import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from docarray import DocList

from marie import requests, safely_encoded
from marie.api import AssetKeyDoc, value_from_payload_or_args
from marie.boxes import PSMode
from marie.executor.pipeline.document_pipeline_executor import PipelineExecutor
from marie.executor.request_util import get_frames_from_docs, parse_parameters
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import torch_gc
from marie.ocr import CoordinateFormat
from marie.ocr.util import get_known_ocr_engines
from marie.pipe.components import (
    burst_frames,
    ocr_frames,
    rotate_frames,
    update_existing_meta,
)
from marie.pipe.llm_pipeline import LLMPipeline
from marie.storage import StorageManager
from marie.utils.asset_util import (
    create_working_dir,
    download_asset,
    restore_assets,
    s3_asset_path,
    split_filename,
    store_assets,
)
from marie.utils.json import load_json_file, store_json_object
from marie.utils.tiff_ops import merge_tiff
from marie.utils.utils import ensure_exists


class DocDeterminationPipelineExecutor(PipelineExecutor):
    """Executor for pipeline document proccessing"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        pipelines: List[dict[str, Any]] = [],
        *args,
        **kwargs,
    ):
        super().__init__(name, device, num_worker_preprocess, storage, **kwargs)
        logger.info("Starting Doc Determination Executor Setup")
        self.ocr_engines = get_known_ocr_engines(self.device.type, "default")
        if pipelines:
            logger.info(f"Pipelines config: {pipelines}")
            self.pipeline = LLMPipeline(pipelines_config=pipelines, device=self.device)

    @requests(on="/document/rotate")
    def rotate_frames(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        :param docs: DocList containing a single AssetKeyDoc.
        :param parameters: Dictionary of request parameters including payload.
        :returns: Dictionary with merge status, runtime_info, and stored assets.
        :raises ConnectionError: If unable to fetch existing assets.
        """
        job_id, ref_id, ref_type, queue_id, payload = parse_parameters(parameters)

        try:
            frames = get_frames_from_docs(docs)
            root_asset_dir = create_working_dir(
                frames, ref_id=ref_id, ref_type=ref_type, job_id=job_id
            )

            s3_root_path = restore_assets(
                ref_id,
                ref_type,
                root_asset_dir,
                overwrite=True,
                # dirs_to_restore=["rotation", "results"],
                full_restore=True,  # full_restore is only option that restores tif
            )
            if s3_root_path is None:
                raise ConnectionError("Unable to collect meta data from")

            metadata = {
                "ref_id": ref_id,
                "ref_type": ref_type,
                "job_id": job_id,
                "pages": f"{len(frames)}",
            }

            metadata["rotation"], any_rotated = rotate_frames(
                ref_id, frames, root_asset_dir
            )

            if any_rotated:
                self.logger.info(f"Re-bursting frames for {ref_id} due to rotation")
                burst_frames(ref_id, frames, root_asset_dir, force=True)

                self.logger.info(f"Merging TIFF pages for {ref_id}")
                _, prefix, _ = split_filename(ref_id)
                img_name = f"{prefix}.tif"
                local_img_path = os.path.join(root_asset_dir, img_name)

                # back up original
                StorageManager.write(
                    local_img_path,
                    f"{s3_asset_path(ref_id, ref_type)}/{prefix}_prerotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif",
                )

                # merge_tiff_frames(frames, os.path.join(root_asset_dir, img_name))
                merge_tiff(
                    os.path.join(root_asset_dir, "burst"),
                    local_img_path,
                    sort_key=lambda name: int(
                        os.path.splitext(os.path.basename(name))[0].rsplit("_", 1)[-1]
                    ),
                )

                store_assets(ref_id, ref_type, root_asset_dir, match_wildcard=img_name)

                # If job is submitted with image not in ref_id directory, need to update it so downstream jobs access rotated tiff
                remote_img_path = f"{s3_asset_path(ref_id, ref_type)}/{img_name}"
                incoming_img_path = docs[0].asset_key
                if incoming_img_path != remote_img_path:
                    StorageManager.write(
                        local_img_path, incoming_img_path, overwrite=True
                    )

            metadata["ocr"] = ocr_frames(
                self.ocr_engines,
                ref_id,
                frames,
                root_asset_dir,
                queue_id=queue_id,
                force=any_rotated,  # Force if pages have been rotated
            )

            meta_path = os.path.join(root_asset_dir, f"{ref_id}.meta.json")
            self.logger.info(f"Storing rotation metadata : {meta_path}")
            if os.path.exists(meta_path):
                metadata = update_existing_meta(
                    load_json_file(meta_path, True), metadata
                )
            store_json_object(metadata, meta_path)
            stored_assets = store_assets(
                ref_id, ref_type, root_asset_dir, match_wildcard="*.json"
            )

            return {
                "status": "success",
                "runtime_info": self.runtime_info,
                "assets": stored_assets,
            }
        finally:
            del frames
            torch_gc()
            MDC.remove("request_id")

    @requests(on="/document/collate")
    def collate(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):
        """
        :param docs: DocList containing a single AssetKeyDoc.
        :param parameters: Dictionary of request parameters including payload.
        :returns: Dictionary with merge status, runtime_info, and stored assets.
        :raises ConnectionError: If unable to fetch existing assets.
        """
        job_id, ref_id, ref_type, queue_id, payload = parse_parameters(parameters)

        try:
            frames = get_frames_from_docs(docs)
            root_asset_dir = create_working_dir(
                frames, ref_id=ref_id, ref_type=ref_type, job_id=job_id
            )

            s3_root_path = restore_assets(
                ref_id, ref_type, root_asset_dir, overwrite=True, full_restore=True
            )
            if s3_root_path is None:
                raise ConnectionError("Unable to collect meta data from")

            meta_path = os.path.join(root_asset_dir, f"{ref_id}.meta.json")
            self.logger.info(f"Retrieving metadata : {meta_path}")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            metadata: Dict[str, Any] = load_json_file(meta_path, True)
            metadata["pages"] = f"{len(frames)}"

            doc_determination_collation(metadata)

            store_json_object(metadata, meta_path)
            stored_assets = store_assets(
                ref_id, ref_type, root_asset_dir, match_wildcard=f"{ref_id}.meta.json"
            )

            return {
                "status": "success",
                "runtime_info": self.runtime_info,
                "assets": stored_assets,
            }
        finally:
            del frames
            MDC.remove("request_id")

    @requests(on=["/document/classify"])
    def handle(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):
        if not self.pipeline:
            raise ValueError("pipeline not initialized.")
        return self.run_llm_pipeline(docs, parameters)

    def run_llm_pipeline(self, docs: DocList[AssetKeyDoc], parameters: dict):
        job_id, ref_id, ref_type, queue_id, payload = parse_parameters(parameters)

        # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
        pms_mode = PSMode.from_value(
            value_from_payload_or_args(payload, "mode", default=str(PSMode.SPARSE))
        )
        coordinate_format = CoordinateFormat.from_value(
            value_from_payload_or_args(
                payload, "format", default=str(CoordinateFormat.XYWH)
            )
        )

        if payload.get("regions", []):
            raise NotImplementedError("Regions is not implemented yet")
        if pms_mode is not PSMode.SPARSE:
            raise NotImplementedError(f"PMS mode `{pms_mode}` is not implemented yet")
        if coordinate_format is not CoordinateFormat.XYWH:
            raise NotImplementedError(
                f"Coordinate format `{coordinate_format}` is not implemented yet"
            )

        pipeline_conf, runtime_conf = self.resolve_runtime_and_pipeline_configs(
            payload.get("features", [])
        )

        pages = get_pipeline_pages(ref_id, ref_type, runtime_conf, pipeline_conf)

        for index, page_set in pages.items():
            if page_set:
                self.logger.info(
                    f"Processing page set {index} with {len(page_set)} pages"
                )

            frames = get_frames_from_docs(docs, page_set)
            root_asset_dir = create_working_dir(
                frames,
                ref_id=ref_id,
                ref_type=ref_type,
                queue_id=queue_id,
                job_id=job_id,
            )
            try:
                metadata = self.pipeline.execute_frames_pipeline(
                    ref_id=ref_id,
                    ref_type=ref_type,
                    frames=frames,
                    root_asset_dir=root_asset_dir,
                    job_id=job_id,
                    queue_id=queue_id,
                    runtime_conf=runtime_conf,
                    pages=page_set,
                )
                if metadata is None:
                    self.logger.error("Metadata is None, this should not happen")
                    raise ValueError("Pipeline Execution Error: Metadata is None")

            except BaseException as error:
                self.logger.error(f"Pipeline error : {error}", exc_info=True)
                raise error
            finally:
                del frames
                torch_gc()

        response = {
            "status": "success",
            "runtime_info": self.runtime_info,
        }
        converted = safely_encoded(lambda x: x)(response)
        MDC.remove("request_id")
        return converted

    def resolve_runtime_and_pipeline_configs(
        self, features
    ) -> tuple[dict[str, Any], dict[Any, Any]]:
        self.logger.debug("Extracting Runtime Config from features list")
        runtime_conf = {}
        pipeline_name = None
        for feature in features:
            if feature.get("type") != "pipeline":
                continue
            pipeline_name = feature.get("name")
            if (
                pipeline_name
                and pipeline_name in self.pipeline.pipeline_configs_dict.keys()
            ):
                runtime_conf = feature
                # If we have multiple pipeline names we want to use the default
                if pipeline_name == self.pipeline.default_pipeline_config["name"]:
                    break

        pipeline_conf = self.pipeline.pipeline_configs_dict.get(
            pipeline_name, self.pipeline.default_pipeline_config
        )
        self.logger.info(f"Resolved Runtime Config: {runtime_conf}")
        return pipeline_conf, runtime_conf


def get_pipeline_pages(
    ref_id,
    ref_type,
    runtime_conf: dict[Any, Any],
    pipeline_conf: dict[str, Any],
) -> dict[int, Optional[list[int]]]:
    config_pages = runtime_conf.get("pages", pipeline_conf.get("pages", None))

    pages = {0: None}
    if isinstance(config_pages, str):
        pages = {0: sorted({int(n) for n in config_pages.split(",")})}
    elif isinstance(config_pages, list):
        if all(isinstance(x, int) for x in config_pages):
            pages = {0: sorted({int(n) for n in config_pages})}
        else:
            # todo there is probably a better way to do this
            ensure_exists("/tmp/marie")
            with tempfile.TemporaryDirectory(dir="/tmp/marie") as temp_asset_dir:
                temp_meta_path = download_asset(
                    ref_id,
                    ref_type,
                    temp_asset_dir,
                    s3_file_path=f"{ref_id}.meta.json",
                )
                if os.path.exists(temp_meta_path):
                    existing_meta = load_json_file(temp_meta_path, True)

            if existing_meta:
                pages = filter_pages_by_classifier_results(config_pages, existing_meta)
            else:
                logger.warning(f"No meta file found for {ref_id}, using all pages")
    elif isinstance(config_pages, dict):
        pages = config_pages
    elif config_pages is not None:
        logger.warning(f"Unexpected pages attr {config_pages}, ignoring")

    logger.info(f"Resolved pipeline pages: {pages}")
    return pages


def filter_pages_by_classifier_results(
    pages_config: list[dict], metadata: dict
) -> dict[int, Optional[list[int]]]:
    try:
        result_pages: dict[int, list[int]] = {}
        i = 0
        for rule in pages_config:
            if rule.get("type") != "classification":
                logger.warning(
                    f"Page filtering is only implemented by classifications, ignoring rule: {rule}"
                )
                continue

            group = rule.get("group")
            method = rule.get("method")

            if method not in ("include", "exclude", "split"):
                logger.warning(
                    f"Unknown page filtering method '{method}', skipping rule: {rule}"
                )
                continue

            filtered_classifications = set(
                str(r) for r in rule.get("classifications", [])
            )
            if not filtered_classifications:
                logger.warning(
                    f"No classifications specified for page filtering, skipping rule: {rule}"
                )
                continue

            # Find existing metadata results to filter on
            target_results = [
                c
                for c in metadata.get("classifications", [])
                if c.get("group") == group
            ] or None
            if not target_results:
                logger.warning(
                    f"Classification '{group}' not found in metadata, skipping page filtering."
                )
                continue

            classification_pages = {}
            for t in target_results:
                classification_pages |= t.get("classification", {}).get("pages", {})

            matched_pages = []
            if method == "exclude":
                # fixme
                matched_pages = [
                    page_results["best"]["page"]
                    for page_results in classification_pages.values()
                    if page_results.get("best", {}).get("classification")
                    not in filtered_classifications
                ]
            elif method in ("include", "split"):
                matched_pages = [
                    page_results["best"]["page"]
                    for page_results in classification_pages.values()
                    if page_results.get("best", {}).get("classification")
                    in filtered_classifications
                ]

            if matched_pages:
                if method in ("include", "exclude"):
                    if not result_pages:
                        result_pages[i] = sorted(matched_pages)
                        i = i + 1
                    else:
                        for k, v in result_pages.items():
                            result_pages[k] = [
                                page for page in v if page in matched_pages
                            ]
                elif method == "split":
                    if not result_pages:
                        result_pages = {
                            i: list(int(k) for k in classification_pages.keys())
                        }

                    split_after = set(matched_pages)
                    out = {}
                    chunk: List[int] = []
                    j = 0
                    for _, group in sorted(result_pages.items()):
                        for x in group:
                            chunk.append(x)
                            if x in split_after:
                                out[j] = chunk
                                j += 1
                                chunk = []
                        if chunk:
                            out[j] = chunk
                            j += 1
                            chunk = []

                    result_pages = out
                    i = j
        return result_pages
    except Exception as e:
        logger.error(
            f"Error while filtering pages with config: {pages_config} Error: {e}"
        )
        logger.warning("Continuing with processing all pages")
        return {0: None}


def doc_determination_collation(meta: Dict[str, Any]) -> Dict[str, Any]:
    documents: List[Dict[str, Any]] = []
    try:
        rotation_pages = meta.get("rotation", {}).get("pages", {})

        # page_number -> medical page classification string (e.g., "EOB")
        medical_by_page: Dict[int, Optional[str]] = {}
        for item in meta.get("classifications", []):
            if item.get("group") != "medical-page-classifier":
                continue
            pages = (item.get("classification") or {}).get("pages") or {}
            for p, pnode in pages.items():
                try:
                    pnum = int(p)
                except (TypeError, ValueError):
                    continue
                best = (pnode or {}).get("best") or {}
                medical_by_page[pnum] = best.get("classification")

        for item in meta.get("classifications", []):
            if item.get("group") != "doc-determination-classifier":
                continue

            cls = item.get("classification", {})
            pages_map = cls.get("pages", {}) or {}

            raw_page_numbers = cls.get("page_numbers")
            if raw_page_numbers is None:
                raw_page_numbers = pages_map.keys()

            page_numbers = sorted({int(p) for p in raw_page_numbers})
            if not page_numbers:
                continue

            # All pages in group should share one doc-determination value.
            labels = set()
            for p in page_numbers:
                node = pages_map.get(str(p), pages_map.get(p, {})) or {}
                best = node.get("best", {}) or {}
                label = best.get("classification")
                if label is None:
                    details = node.get("details", []) or []
                    if details:
                        label = details[0].get("classification")
                if label is not None:
                    labels.add(label)

            if len(labels) > 1:
                raise ValueError(
                    f"Inconsistent classifications in doc-determination-classifier group: {labels}"
                )
            doc_label = next(iter(labels), None)

            # Fill missing pages in range (e.g., [2,3,5] -> [2,3,4,5])
            start_page, end_page = page_numbers[0], page_numbers[-1]
            collated_pages = []
            for page_num in range(start_page, end_page + 1):
                rot = (
                    rotation_pages.get(str(page_num), rotation_pages.get(page_num, {}))
                    or {}
                )
                collated_pages.append(
                    {
                        "page": page_num,
                        "rotation": rot.get("rotate"),
                        "medical-page-classification": medical_by_page.get(page_num),
                    }
                )

            documents.append(
                {
                    "doc-classification": doc_label,
                    "page-count": len(collated_pages),
                    "pages": collated_pages,
                }
            )

        logger.info(f"Final document count: {len(documents)}")
    except Exception as e:
        logger.error(f"Error during document collation: {e}")
        meta.setdefault("doc_determination_collation", {})["Error"] = e

    meta["doc_determination_collation"] = {
        "doc_count": len(documents),
        "docs": documents,
    }

    return meta
