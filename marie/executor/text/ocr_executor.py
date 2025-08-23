import os
from typing import Any, Optional

from docarray import DocList

from marie import requests, safely_encoded
from marie.api import (
    AssetKeyDoc,
    get_frames_from_docs,
    get_payload_features,
    parse_parameters,
)
from marie.executor.extract.document_extractor_executor import DocumentExtractExecutor
from marie.executor.extract.util import create_working_dir
from marie.ocr.util import get_known_ocr_engines
from marie.pipe.components import (
    asset_exists,
    ocr_frames,
    s3_asset_path,
    split_filename,
    store_assets,
)
from marie.utils.image_utils import hash_frames_fast


class OCRExecutor(DocumentExtractExecutor):
    """Executor for OCR only processing"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        engine: str = "default",
        **kwargs,
    ):
        super().__init__(name, device, num_worker_preprocess, storage, **kwargs)
        self.logger.info(f"Setup OCR engine: {engine}")
        self.ocr_engines = get_known_ocr_engines(self.device.type, engine)

    @requests(on="/document/ocr")
    def handle_ocr(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):
        job_id, ref_id, ref_type, queue_id, payload = parse_parameters(parameters)
        frames = get_frames_from_docs(docs)
        ref_id = hash_frames_fast(frames) if ref_id is None else ref_id
        ref_type = "extract" if ref_type is None else ref_type
        root_asset_dir = create_working_dir(frames)

        self.logger.info(f"Starting OCR for ref_id: {ref_id}")

        # Determine OCR force regeneration from request features
        features = get_payload_features(payload, f_type="extract", name="ocr")
        force_regen = any(True for feature in features if feature.get("force", False))

        if not force_regen:
            _, prefix, _ = split_filename(ref_id)
            ocr_path = f"results/{prefix}.json"
            if asset_exists(ref_id, ref_type, s3_file_path=ocr_path):
                self.logger.info(f"OCR results already exist for ref_id: {ref_id}")
                ocr_path = os.path.join(s3_asset_path(ref_id, ref_type), ocr_path)
                return safely_encoded(lambda x: x)(
                    {
                        "status": "success",
                        "runtime_info": self.runtime_info,
                        "assets": [ocr_path],
                    }
                )

            local_ocr_path = os.path.join(root_asset_dir, ocr_path)
            if os.path.exists(local_ocr_path):
                self.logger.info(f"Removing stale cache: {local_ocr_path}")
                os.remove(local_ocr_path)

        ocr = ocr_frames(
            self.ocr_engines, ref_id, frames, root_asset_dir, force=force_regen
        )
        self.logger.info(f"OCR-ed pages: {len(ocr)}")

        stored_assets = store_assets(
            ref_id, ref_type, root_asset_dir, match_wildcard="results/*.json"
        )
        self.logger.debug(f"Stored assets: {stored_assets}")

        return safely_encoded(lambda x: x)(
            {
                "status": "success",
                "runtime_info": self.runtime_info,
                "assets": stored_assets,
            }
        )
