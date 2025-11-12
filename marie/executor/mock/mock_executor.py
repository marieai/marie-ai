import asyncio
import os
import random
import time
from typing import Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import (
    setup_torch_optimizations,
)
from marie.utils.network import get_ip_address


class IntegrationExecutorMock(MarieExecutor):
    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        pipeline: Optional[dict[str, any]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        process_time: float = 3.0,
        failure_rate: float = 0.0,
        failure_mode: str = "exception",
        **kwargs,
    ):
        """
        Mock executor for integration testing with configurable behavior.

        :param device: 'cpu' or 'cuda'. Default is None, which auto-detects the device.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param storage: Storage configuration dictionary with 'psql' settings for asset tracking.
        :param pipeline: Pipeline configuration dictionary.
        :param dtype: inference data type, if None defaults to torch.float32 if device == 'cpu' else torch.float16.
        :param process_time: Base processing time in seconds. Default is 3.0. Can be overridden per request.
        :param failure_rate: Probability of failure (0.0 to 1.0). Default is 0.0 (no failures).
        :param failure_mode: Type of failure to simulate: 'exception', 'timeout', 'random'. Default is 'exception'.
        """
        kwargs['storage'] = storage
        super().__init__(**kwargs)

        logger.info(f"Starting mock executor : {time.time()}")
        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Storage config: {storage}")
        logger.info(f"Pipeline config: {pipeline}")
        logger.info(f"Device : {device}")
        logger.info(f"Num worker preprocess : {num_worker_preprocess}")
        logger.info(f"Process time : {process_time}s")
        logger.info(f"Failure rate : {failure_rate}")
        logger.info(f"Failure mode : {failure_mode}")
        logger.info(f"Kwargs : {kwargs}")

        setup_torch_optimizations()

        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not use_cuda:
            device = "cpu"
        self.device = device

        # Mock behavior configuration
        self.process_time = process_time
        self.failure_rate = max(0.0, min(1.0, failure_rate))  # Clamp between 0 and 1
        self.failure_mode = failure_mode

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": kwargs.get("runtime_args", {}).get("name", "not_defined"),
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
            "device": self.device.__str__() if self.device is not None else "",
            "process_time": self.process_time,
            "failure_rate": self.failure_rate,
            "failure_mode": self.failure_mode,
        }

        logger.info(f"Runtime info: {self.runtime_info}")
        logger.info(f"Pipeline : {pipeline}")

    # @requests(on="/document/status")
    # def status(self, parameters, **kwargs):
    #     use_cuda = torch.cuda.is_available()
    #     print(f"{use_cuda=}")
    #     return {"index": "complete", "use_cuda": use_cuda}
    #
    # @requests(on="/document/validate")
    # def validate(self, parameters, **kwargs):
    #     return {"valid": True}

    @requests(on="/document/process")
    async def func_extract(
        self,
        docs: DocList[AssetKeyDoc],
        parameters=None,
        *args,
        **kwargs,
    ):
        """
        Process documents with configurable processing time and failure simulation.

        Parameters can override executor-level settings via the 'parameters' dict:
        - 'process_time': Override the base processing time
        - 'failure_rate': Override the failure rate for this request
        - 'failure_mode': Override the failure mode for this request
        - 'force_fail': If True, force a failure regardless of failure_rate
        """
        if parameters is None:
            parameters = {}

        self.logger.info(f"func called : {len(docs)}, {parameters}")

        # Get processing time (allow per-request override)
        process_time = parameters.get("process_time", self.process_time)

        # Add randomness if requested (for more realistic simulation)
        if parameters.get("randomize_time", False):
            min_time = process_time * 0.5
            max_time = process_time * 1.5
            process_time = random.uniform(min_time, max_time)

        # Check if we should fail this request

        failure_rate = parameters.get("failure_rate", self.failure_rate)
        failure_mode = parameters.get("failure_mode", self.failure_mode)
        force_fail = parameters.get("force_fail", False)

        should_fail = force_fail or (random.random() < failure_rate)

        if should_fail:
            self.logger.warning(f"Simulating failure (mode: {failure_mode})")

            if failure_mode == "exception":
                raise RuntimeError(
                    f"Mock failure: Simulated exception in {self.runtime_info['instance_name']}"
                )
            elif failure_mode == "timeout":
                # Simulate a timeout by sleeping much longer
                self.logger.warning(
                    "Simulating timeout by sleeping for extended period"
                )
                await asyncio.sleep(process_time * 10)
                raise TimeoutError(
                    f"Mock failure: Simulated timeout in {self.runtime_info['instance_name']}"
                )
            elif failure_mode == "random":
                # Randomly choose between different failure types
                failure_types = [
                    RuntimeError("Mock failure: Random runtime error"),
                    ValueError("Mock failure: Random value error"),
                    ConnectionError("Mock failure: Random connection error"),
                ]
                raise random.choice(failure_types)

        # Normal processing - simulate work
        self.logger.info(f"Processing for {process_time:.2f} seconds : {time.time()}")
        await asyncio.sleep(process_time)
        self.logger.info(f"Processing complete : {time.time()}")

        result = {
            "parameters": parameters,
            "data": "Data reply",
            "processed_docs": len(docs),
            "process_time": process_time,
            "executor": self.runtime_info["instance_name"],
        }

        # Extract optional DAG tracking parameters
        job_id = parameters.get("job_id")
        dag_id = parameters.get("dag_id")
        node_task_id = parameters.get("node_task_id")
        partition_key = parameters.get("partition_key")

        # Persist results and track assets if enabled
        if job_id:
            self.persist(
                results=result,
                job_id=job_id,
                dag_id=dag_id,
                node_task_id=node_task_id,
                partition_key=partition_key,
            )

        print(f"Mock executor results : {result}")
        return result

    def persist(
        self,
        results: any,
        job_id: Optional[str] = None,
        dag_id: Optional[str] = None,
        node_task_id: Optional[str] = None,
        partition_key: Optional[str] = None,
    ) -> None:
        """Persist results and optionally track assets"""
        from typing import Any

        from marie.api.docs import StorageDoc

        def _tags(index: int, ftype: str, checksum: str):
            return {
                "action": "mock_process",
                "index": index,
                "type": ftype,
                "ttl": 48 * 60,
                "checksum": checksum,
                "runtime": self.runtime_info,
            }

        # Legacy storage that persist documents directly
        if self.storage_enabled:
            docs = DocList[StorageDoc](
                [
                    StorageDoc(
                        content=results,
                        tags=_tags(-1, "metadata", job_id or "unknown"),
                    )
                ]
            )

            self.store(
                ref_id=job_id or "unknown",
                ref_type="mock_process",
                store_mode="content",
                docs=docs,
            )

        # Asset tracking
        if self.asset_tracking_enabled and job_id:
            import hashlib
            import json

            from marie.assets import AssetTracker

            assets = []

            # Create a random number of mock assets between 1 and 5
            n_assets = random.randint(1, 5)

            for i in range(n_assets):
                # Create per-asset metadata
                result_data = {
                    "processed_docs": results.get("processed_docs", 0),
                    "process_time": results.get("process_time", 0),
                    "executor": results.get("executor", "unknown"),
                    "asset_index": i,
                }
                result_bytes = json.dumps(result_data).encode("utf-8")

                # Compute version using upstream versions
                upstream_versions = self._get_upstream_versions(dag_id, node_task_id)
                version = AssetTracker.compute_asset_version(
                    payload_bytes=result_bytes,
                    code_fingerprint=getattr(self, "code_version", "unknown"),
                    prompt_fingerprint=f"mock-executor-v1-{i}",
                    upstream_versions=upstream_versions,
                )

                assets.append(
                    {
                        "asset_key": f"mock/processed/{job_id}/{i}",
                        "version": version,
                        "kind": "json",
                        "size_bytes": len(result_bytes),
                        "checksum": hashlib.sha256(result_bytes).hexdigest(),
                        "metadata": result_data,
                    }
                )

            # Record materializations
            if assets:
                try:
                    upstream = self._get_upstream_asset_tuples(dag_id, node_task_id)

                    # Note: This is a blocking call that returns a Future
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
                        f"Recorded {len(assets)} asset materializations for job {job_id}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to record asset materializations: {e}", exc_info=True
                    )
