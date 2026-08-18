import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
from marie.pipe.components import burst_frames
from marie.pipe.llm_pipeline import LLMPipeline
from marie.utils.asset_util import (
    create_working_dir,
    download_asset,
    restore_assets,
    store_assets,
)
from marie.utils.json import load_json_file, store_json_object
from marie.utils.tiff_ops import merge_tiff_frames_with_splits_ifd
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
            stored_assets = []
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

            # Add doc determination collation to metadata
            metadata: Dict[str, Any] = load_json_file(meta_path, True)
            metadata["pages"] = f"{len(frames)}"
            collation = doc_determination_collation(metadata)
            metadata["doc_determination_collation"] = collation
            store_json_object(metadata, meta_path)
            stored_assets.append(
                store_assets(
                    ref_id,
                    ref_type,
                    root_asset_dir,
                    match_wildcard=f"{ref_id}.meta.json",
                )
            )

            # Split all metadata files
            if collation.get("doc_count", 0) > 1:
                burst_frames(ref_id, frames, root_asset_dir, force=False)
                split_assets(collation, root_asset_dir, ref_id)
                stored_assets.append(
                    store_assets(
                        ref_id, ref_type, root_asset_dir, match_wildcard="splits/**/*"
                    )
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
            min_conf = rule.get("min_conf", 0)

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

            matched_pages = [
                page_results["best"]["page"]
                for page_results in classification_pages.values()
                if page_results.get("best", {}).get("classification")
                in filtered_classifications
                and page_results.get("best", {}).get("score", 0) >= min_conf
            ]

            if not result_pages:
                # NOTE: Pages which were skipped by the classification we are filtering on will be filtered out in the result pages
                result_pages = {i: list(int(k) for k in classification_pages.keys())}

            if matched_pages:
                if method == "include":
                    for k, v in result_pages.items():
                        result_pages[k] = [page for page in v if page in matched_pages]
                elif method == "exclude":
                    for k, v in result_pages.items():
                        result_pages[k] = [
                            page for page in v if page not in matched_pages
                        ]
                elif method == "split":
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
        rotation_threshold = meta.get("rotation", {}).get("threshold", 0)
        rotation_pages = meta.get("rotation", {}).get("pages", {})

        _all_pages = [int(x) for x in rotation_pages.keys()]

        classifications_by_group = defaultdict(list)
        for item in meta.get("classifications", []):
            classifications_by_group[item.get("group")].append(
                item.get("classification") or {}
            )

        # page_number -> medical page classification string (e.g., "EOB")
        medical_by_page: Dict[int, Optional[str]] = {}
        for item in classifications_by_group["medical-page-classifier"]:
            pages = item.get("pages") or {}
            for p, pnode in pages.items():
                best = (pnode or {}).get("best") or {}
                medical_by_page[int(p)] = best.get("classification")

        doc_determinations = classifications_by_group["doc-determination-classifier"]
        doc_determinations.sort(
            key=lambda x: min({int(p) for p in x.get("pages", {}).keys()} or {0})
        )
        start_doc_page, last_page = min(_all_pages), max(_all_pages)
        for idx, item in enumerate(doc_determinations):
            pages_map = item.get("pages", {}) or {}
            page_numbers = sorted({int(p) for p in pages_map.keys()})
            if not page_numbers:
                continue

            # All pages in group should share one doc-determination value.
            labels = {
                (node.get("best") or {}).get("classification")
                for node in pages_map.values()
            }
            labels.discard(None)

            if len(labels) > 1:
                raise ValueError(
                    f"Inconsistent classifications in doc-determination-classifier group: {labels}"
                )
            doc_label = next(iter(labels), None)

            # Fill missing pages in range (e.g., [2,3,5] -> [2,3,4,5])
            # Make sure first and last pages are accounted for (e.g., if FIRST classification is [2,3,5] -> [1,2,3,4,5])
            end_doc_page = (
                last_page if idx == len(doc_determinations) - 1 else page_numbers[-1]
            )

            collated_pages = []
            for new_page, page_num in enumerate(
                range(start_doc_page, end_doc_page + 1)
            ):
                rot = (
                    rotation_pages.get(str(page_num), rotation_pages.get(page_num, {}))
                    or {}
                ).get("rotate", 0)
                collated_pages.append(
                    {
                        "page": page_num,
                        "new_page": new_page,
                        "rotation": rot if rot >= rotation_threshold else 0,
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

            start_doc_page = end_doc_page + 1

        logger.info(f"Final document count: {len(documents)}")
    except Exception as e:
        logger.error(f"Error during document collation: {e}")
        return {"error": str(e)}

    return {"doc_count": len(documents), "docs": documents}


def split_meta_json(
    meta: Dict[str, Any] | List[Dict[str, Any]], pages: List[int]
) -> None | dict[str, Any] | list[dict[str, Any]]:
    """

    :param meta: metadata
    :param pages: list of pages in the order they should appear in split meta
    :return:
    """

    def split_ocr(
        ocr: list[dict[str, Any]], new_pages: dict[int, int]
    ) -> list[dict[str, dict[str, int]]]:
        return [
            {
                **x,
                "meta": {
                    **x["meta"],
                    "page": n,
                },
            }
            for x in ocr
            if (n := new_pages.get(int(x.get("meta", {}).get("page")))) is not None
        ]

    def split_rotation(rotation: dict[str, Any], new_pages: dict[int, int]):
        threshold_ = rotation["threshold"]
        pages_ = {
            str(n): v
            for (k, v) in rotation["pages"].items()
            if (n := new_pages.get(int(k))) is not None
        }
        any_rotated_ = any(
            x["rotate"] > 0 and x["orientation_conf"] > threshold_
            for x in pages_.values()
        )
        return {
            "any_rotated": any_rotated_,
            "threshold": threshold_,
            "pages": pages_,
        }

    def split_classifications(
        classifications,
        new_pages: dict[int, int],
    ):
        _classifications = []
        old_pages = set(new_pages.keys())
        for item in classifications:
            cls = item.get("classification") or {}
            classification_pages = cls.get("pages") or {}
            classification_page_numbers = cls.get("page_numbers") or []
            if {int(x) for x in classification_pages.keys()} & old_pages:
                _classifications.append(
                    {
                        **item,
                        "classification": {
                            **cls,
                            "page_numbers": [
                                n
                                for x in classification_page_numbers
                                if (n := new_pages.get(int(x))) is not None
                            ],
                            "pages": {
                                str(n): {
                                    "details": [{**d, "page": n} for d in v["details"]],
                                    "best": {**v["best"], "page": n},
                                }
                                for (k, v) in classification_pages.items()
                                if (n := new_pages.get(int(k))) is not None
                            },
                        },
                    }
                )

        return _classifications

    def split_indexes(
        indexes,
        new_pages: dict[int, int],
    ):
        # todo implement
        return indexes

    try:
        new_pages = {p: i for i, p in enumerate(pages)}

        if isinstance(meta, list):
            # OCR JSON
            return split_ocr(meta, new_pages)
        elif isinstance(meta, dict):
            if meta.get("any_rotated"):
                # ROTATION JSON
                return split_rotation(meta, new_pages)

            _meta = {}
            _meta["pages"] = len(pages)
            _meta["job_id"] = meta.get("job_id", "")
            _meta["pipeline"] = meta.get("pipeline", "")
            _meta["parent_ref_id"] = meta.get("ref_id", "")
            _meta["parent_pages"] = pages
            _meta["ref_type"] = meta.get("ref_type", "")

            # rotation
            rotation = meta.get("rotation", {})
            if rotation:
                split_rotation(rotation, new_pages)

            # classifications
            classifications = meta.get("classifications", [])
            if classifications:
                _meta["classifications"] = split_classifications(
                    classifications, new_pages
                )

            # indexes
            indexes = meta.get("indexers", [])
            if indexes:
                _meta["indexers"] = split_indexes(indexes, new_pages)

            # ocr
            ocr = meta.get("ocr", [])
            if ocr:
                _meta["ocr"] = split_ocr(ocr, new_pages)

            return _meta
        else:
            logger.error(f"Unexpected metadata type: {type(meta)}")
    except Exception as e:
        logger.error(f"Error during document splitting: {e}")
    return None


def split_all_meta_jsons(
    splits: List[Dict[str, Any]],
    input_dir: str,
    output_dir: str,
    filename_generator: Optional[Callable[[int, int, int], str]] = None,
    glob: str = "*.json",
):
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(exist_ok=True, parents=True)
    in_dir_path = Path(input_dir)

    filename_generator = filename_generator or (
        lambda subpath, start_page, end_page: f"{start_page}-{end_page}/{subpath}"
    )

    jsons = list(in_dir_path.rglob(glob, case_sensitive=False))
    for file_path in jsons:
        sub_path = file_path.relative_to(in_dir_path)
        if sub_path.parts[0] == "splits":
            continue

        logger.info(f"Splitting file: {sub_path}")
        metadata: Dict[str, Any] = load_json_file(file_path, True)

        for idx, item in enumerate(splits):
            pages = [int(x.get("page")) for x in item.get("pages", [])]
            meta_split = split_meta_json(metadata, pages)

            if meta_split:
                output_name = filename_generator(sub_path, min(pages), max(pages))
                meta_path = Path(
                    output_name
                    if os.path.isabs(output_name)
                    else os.path.join(str(output_dir), output_name)
                )

                meta_path.parent.mkdir(exist_ok=True, parents=True)
                store_json_object(meta_split, meta_path)


def split_assets(collation: dict[str, Any], root_asset_dir: str, ref_id: str):
    split_output_dir = os.path.join(root_asset_dir, "splits")
    split_docs = collation.get("docs", [])

    if os.path.exists(split_output_dir):
        shutil.rmtree(split_output_dir)

    # Split all JSON files
    split_all_meta_jsons(
        split_docs,
        root_asset_dir,
        split_output_dir,
        filename_generator=lambda subpath, start_page, end_page: f"{start_page}-{end_page}/{str.replace(str(subpath), ref_id, '__CHILD_REFID__')}",
        glob=f"*{ref_id}*.json",
    )

    # Split tiffs
    indices = [x["pages"][0]["page"] for x in split_docs]
    burst_dir = os.path.join(root_asset_dir, "burst")
    if os.path.exists(burst_dir):
        merge_tiff_frames_with_splits_ifd(
            burst_dir,
            indices,
            split_output_dir,
            sort_key=lambda name: int(
                os.path.splitext(os.path.basename(name))[0].rsplit("_", 1)[-1]
            ),
            filename_generator=lambda _, start_page, end_page: f"{start_page}-{end_page}/__CHILD_REFID__.tif",
        )
