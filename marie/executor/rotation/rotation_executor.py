import os
from datetime import datetime
from typing import Any, Optional

import torch
from docarray import DocList
from tesserocr import PSM, PyTessBaseAPI

from marie.api import AssetKeyDoc
from marie.executor.marie_executor import MarieExecutor
from marie.executor.request_util import get_frames_from_docs, parse_parameters
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import initialize_device_settings, torch_gc
from marie.ocr.util import get_known_ocr_engines
from marie.pipe.components import (
    burst_frames,
    ocr_frames,
    rotate_frames,
    update_existing_meta,
)
from marie.runtime import requests
from marie.storage import StorageManager
from marie.utils.asset_util import (
    create_working_dir,
    restore_assets,
    s3_asset_path,
    split_filename,
    store_assets,
)
from marie.utils.json import load_json_file, store_json_object
from marie.utils.network import get_ip_address
from marie.utils.tiff_ops import merge_tiff_frames_ifd


class RotationExecutor(MarieExecutor):
    """Executor for image rotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = "cuda",
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        kwargs["storage"] = storage
        super().__init__(**kwargs)

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.warning(f"Device is not specified, using {device} as default device")
        use_cuda = device == "cuda"

        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Storage config: {storage}")
        logger.info(f"Device : {device}")
        logger.info(f"Kwargs : {kwargs}")

        instance_name = (
            kwargs.get("runtime_args", {}).get("name", "not_defined")
            if kwargs is not None
            else "not_defined"
        )

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": instance_name,
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
        }

        resolved_devices, _ = initialize_device_settings(
            devices=[device], use_cuda=use_cuda, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]
        self.ocr_engines = get_known_ocr_engines(self.device.type, "default")

        self.logger.info("Initializing PyTessBaseAPI...")
        self.rotation_api = PyTessBaseAPI(lang="osd", psm=PSM.OSD_ONLY)

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
                ref_id, frames, root_asset_dir, api=self.rotation_api
            )

            if any_rotated:
                self.logger.info(f"Re-bursting frames for {ref_id} due to rotation")
                burst_frames(ref_id, frames, root_asset_dir, force=True)

                self.logger.info(f"Merging TIFF pages for {ref_id}")
                _, prefix, _ = split_filename(ref_id)
                img_name = f"{prefix}.tif"
                local_img_path = os.path.join(root_asset_dir, img_name)

                # back up original
                if os.path.exists(local_img_path):
                    StorageManager.write(
                        local_img_path,
                        f"{s3_asset_path(ref_id, ref_type)}/{prefix}_prerotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif",
                    )

                merge_tiff_frames_ifd(
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
