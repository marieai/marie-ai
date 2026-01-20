import os
import random
import time
from typing import Optional, Union

import torch
from docarray import DocList
from omegaconf import OmegaConf

from marie.api.docs import AssetKeyDoc
from marie.constants import __config_dir__
from marie.excepts import RuntimeTerminated
from marie.executor.asset_util import prepare_asset_directory
from marie.executor.extract.util import layout_config
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin
from marie.extract.annotators.types import AnnotatorClassType
from marie.extract.readers.meta_reader.meta_reader import MetaReader
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import torch_gc
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.json import load_json_file
from marie.utils.network import get_ip_address


class DocumentAnnotatorExecutor(MarieExecutor, StorageMixin):
    """Executor for document annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        llm_tracking: dict[str, any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        kwargs['storage'] = storage
        kwargs['llm_tracking'] = llm_tracking
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Storage config: {storage}")
        logger.info(f"Device : {device}")
        logger.info(f"Num worker preprocess : {num_worker_preprocess}")
        logger.info(f"Kwargs : {kwargs}")
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        instance_name = "not_defined"
        if kwargs is not None:
            if "runtime_args" in kwargs:
                instance_name = kwargs.get("runtime_args").get("name", "not_defined")

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": instance_name,
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
        }

        self.storage_enabled = False
        self.asset_tracking_enabled = False
        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            # Check if asset tracking is enabled in config
            asset_tracking = sconf.get("asset_tracking_enabled", False)
            self.setup_storage(
                storage_enabled=sconf.get("enabled", False),
                storage_conf=sconf,
                asset_tracking_enabled=asset_tracking,
            )

        # Setup LLM tracking if configured (for executor process)
        if llm_tracking is not None and llm_tracking.get("enabled", False):
            self._setup_llm_tracking(llm_tracking, storage)

        self.root_config_dir = os.path.join(__config_dir__, "extract")
        self.logger.info(f"root_config_dir: {self.root_config_dir}")

    # @requests(on="/default")
    # def default(self,
    #         docs: DocList[AssetKeyDoc],
    #         parameters: dict, **kwargs):
    #     print('===================== DEFAULT =====================')
    #     self.logger.warning(f"Default endpoint called")
    #     print(docs)
    #     print(parameters)
    #     print(kwargs)
    #     raise NotImplementedError(
    #         'Invalid(/default) endpoint have been called, ensure your config are correct'
    #     )

    def _setup_request(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        if len(docs) == 0:
            return {"error": "empty payload"}
        if len(docs) > 1:
            return {"error": "expected single document"}

        if parameters is None or "job_id" not in parameters:
            self.logger.warning(f"Job ID is not present in parameters")
            raise ValueError("Job ID is not present in parameters")

        job_id = parameters.get("job_id")
        MDC.put("request_id", job_id)

        self.logger.info("processing request parameters")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

    def _setup_llm_tracking(
        self, llm_tracking_config: dict, storage_config: dict = None
    ) -> None:
        """
        Configure LLM tracking for this executor process.

        Executors run in separate processes (via SPAWN) and don't inherit the
        gateway's configuration. This method calls configure_from_yaml() to
        initialize LLM tracking settings in the executor process.

        Args:
            llm_tracking_config: The llm_tracking section from YAML config
            storage_config: Optional storage section for shared S3 config
        """
        try:
            from marie.llm_tracking.config import configure_from_yaml

            configure_from_yaml(llm_tracking_config, storage_config)
            self.logger.info("LLM tracking configured for executor process")
        except Exception as e:
            self.logger.warning(f"Failed to configure LLM tracking: {e}")

    async def _process_annotation_request(
        self,
        docs: DocList[AssetKeyDoc],
        parameters: dict,
        annotator_class: AnnotatorClassType,
        *args,
        **kwargs,
    ):
        """
        Common processing method for document annotation requests

        :param docs: Documents to process
        :param parameters: Request parameters
        :param annotator_class: The annotator class to use (LLMAnnotator or LLMTableAnnotator)
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: Response dictionary
        """

        if False:
            # Simulate a long-running process with a random chance of cancellation
            self.logger.info(
                "Simulating long-running process with a random chance of cancellation"
            )
            self.logger.info(
                "========================= SLEEPING ========================="
            )
            # if random.random() < 0.2:  # 20% chance to simulate async cancellation
            #     raise asyncio.CancelledError("Simulated async task cancellation")

            sec = random.randint(1, 2)
            # sec = 0
            # time.sleep(sec)  # this will trigger
            for i in range(sec):
                # await asyncio.sleep(1)
                time.sleep(1)
                self.logger.info(f"Sleeping... {i + 1}/{sec} seconds elapsed")

            return {'status': 'success', 'message': 'Documents annotated successfully'}

        self._setup_request(docs, parameters)

        # load documents from specified document asset key
        doc: AssetKeyDoc = docs[0]
        self.logger.info(f"doc.asset_key = {doc.asset_key}")
        docs, local_downloaded_s3_path = docs_from_asset(
            doc.asset_key, doc.pages, return_file_path=True
        )
        frames = frames_from_docs(docs)

        try:
            job_id = parameters.get("job_id")
            ref_id = parameters.get("ref_id")
            ref_type = parameters.get("ref_type")
            payload = parameters.get("payload")
            op_params = payload.get(
                "op_params"
            )  # These are operator parameters (Layout, Config Key, etc.)

            op_key = op_params.get('key')
            op_layout = op_params.get('layout')

            self.logger.info(f"Executing operation with params : {op_params}")
            self.logger.info(f"Extracted op_key: {op_key}")
            self.logger.info(f"Extracted op_layout: {op_layout}")

            cfg = layout_config(self.root_config_dir, op_layout)

            annotator_conf = None
            for annotator in cfg.annotators:
                if annotator == op_key:
                    annotator_conf = cfg.annotators[annotator]
                    # we need to set the name of the annotator as they are keys in conf
                    annotator_conf['name'] = annotator
                    break

            if annotator_conf is None:
                raise ValueError(f"Invalid annotator key: {op_key}")
            # remove any dependencies on OmegaConf to avoid issues with index access
            annotator_conf = OmegaConf.to_container(annotator_conf, resolve=True)

            root_asset_dir, frames_dir, metadata_file = prepare_asset_directory(
                frames=frames,
                local_path=local_downloaded_s3_path,
                ref_id=ref_id,
                ref_type=ref_type,
                logger=self.logger,
            )
            self.logger.info(f"root_asset_dir = {root_asset_dir}")

            # self.logger.info(f"Downloaded assets to {metadata_file}")
            metadata = load_json_file(metadata_file)
            unstructured_meta = {
                'ref_id': ref_id,
                'ref_type': ref_type,
                'job_id': job_id,
                'source_metadata': metadata,
            }

            doc: UnstructuredDocument = MetaReader.from_data(
                frames=frames,
                ocr_meta=metadata["ocr"],
                unstructured_meta=unstructured_meta,
            )
            self.logger.info(f"Doc : {doc}")
            self.logger.info(f"Doc page_count: {doc.page_count}")

            annotator = annotator_class(
                working_dir=root_asset_dir,
                annotator_conf=annotator_conf,
                layout_conf={
                    "layout_id": op_layout,
                },
            )

            await annotator.aannotate(doc, frames)
            del annotator

            # Extract DAG tracking parameters for asset tracking
            dag_id = parameters.get("dag_id")
            node_task_id = parameters.get("node_task_id")
            partition_key = parameters.get("partition_key")

            # Record asset materializations if enabled
            self._record_annotation_assets(
                job_id=job_id,
                ref_id=ref_id,
                ref_type=ref_type,
                op_key=op_key,
                op_layout=op_layout,
                page_count=doc.page_count,
                dag_id=dag_id,
                node_task_id=node_task_id,
                partition_key=partition_key,
            )

            response = {
                "status": "success",
                "runtime_info": self.runtime_info,
                "error": None,
            }
            return response
        except BaseException as error:
            # If GPU failure is detected, escalate to RuntimeTerminated so the worker can restart
            try:
                if hasattr(self, "_is_gpu_failure") and self._is_gpu_failure(error):
                    self._raise_runtime_terminated(
                        "GPU failure during annotation", error
                    )
            except RuntimeTerminated:
                # Re-raise to let the Runtime handle termination
                raise
            except Exception:
                pass

            self.logger.error(f"Extract error : {error}", exc_info=True)
            msg = "inference exception"
            if self.show_error:
                msg = (str(error),)
            return {
                "status": "error",
                "runtime_info": self.runtime_info,
                "error": msg,
            }
        finally:
            torch_gc()
            MDC.remove("request_id")

    def _record_annotation_assets(
        self,
        job_id: str,
        ref_id: str,
        ref_type: str,
        op_key: str,
        op_layout: str,
        page_count: int,
        dag_id: Optional[str] = None,
        node_task_id: Optional[str] = None,
        partition_key: Optional[str] = None,
    ) -> None:
        """Record asset materializations for annotation results"""

        if not self.asset_tracking_enabled or not job_id:
            return

        try:
            import hashlib
            import json

            from marie.assets import AssetTracker

            # Create a fingerprint of the annotation operation
            annotation_fingerprint = f"{op_key}:{op_layout}:{ref_id}:{page_count}"
            fingerprint_bytes = annotation_fingerprint.encode("utf-8")

            # Compute version based on operation parameters
            upstream_versions = self._get_upstream_versions(dag_id, node_task_id)
            version = AssetTracker.compute_asset_version(
                payload_bytes=fingerprint_bytes,
                code_fingerprint=getattr(self, "code_version", "unknown"),
                prompt_fingerprint=f"{op_key}:{op_layout}",
                upstream_versions=upstream_versions,
            )

            assets = [
                {
                    "asset_key": f"annotation/{op_key}",
                    "version": version,
                    "kind": "annotation",
                    "size_bytes": len(fingerprint_bytes),
                    "checksum": hashlib.sha256(fingerprint_bytes).hexdigest(),
                    "metadata": {
                        "op_key": op_key,
                        "op_layout": op_layout,
                        "ref_id": ref_id,
                        "ref_type": ref_type,
                        "page_count": page_count,
                    },
                }
            ]

            # Record materializations
            upstream = self._get_upstream_asset_tuples(dag_id, node_task_id)
            self.asset_tracker.record_materializations(
                storage_event_id=None,
                assets=assets,
                job_id=job_id,
                dag_id=dag_id,
                node_task_id=node_task_id,
                partition_key=partition_key,
                upstream_assets=upstream,
            )
            self.logger.debug(
                f"Recorded annotation asset materialization for job {job_id}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to record asset materializations: {e}", exc_info=True
            )
