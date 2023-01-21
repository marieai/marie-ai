import os
from typing import Dict, Union, Optional

import torch
from docarray import DocumentArray, Document

from marie import Executor, requests, safely_encoded
from marie.executor.mixin import StorageMixin
from marie.logging.logger import MarieLogger
from marie.overlay.overlay import OverlayProcessor
from marie.utils.docs import array_from_docs
from marie.utils.image_utils import (
    hash_frames_fast,
    convert_to_bytes,
)
from marie.utils.utils import ensure_exists


class OverlayExecutor(Executor, StorageMixin):
    """Executor for creating text overlays."""

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        storage_enabled: bool = False,
        storage_conf: Dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        work_dir = ensure_exists("/tmp/form-segmentation")
        self.overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=use_cuda)

        print(f"storage_conf  = {storage_conf}")
        self.logger.info(f"Storage enabled: {storage_enabled}")
        self.setup_storage(storage_enabled, storage_conf)

    @requests(on="/overlay/segment")
    @safely_encoded
    def segment(
        self, docs: DocumentArray, parameters: Optional[Dict] = None, *args, **kwargs
    ):
        """
        Segment document

        EXAMPLE USAGE

            As Executor

            .. code-block:: python

                filename = img_path.split("/")[-1].replace(".png", "")
                checksum = hash_file(img_path)
                docs = docs_from_file(img_path)

                parameters = {
                    "ref_id": filename,
                    "ref_type": "filename",
                    "checksum": checksum,
                    "img_path": img_path,
                }

                results = executor.segment(docs, parameters)


        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        self.logger.info("Starting segment request")

        try:
            frames = array_from_docs(docs)
            self.logger.info(f"Processing total frames : {len(frames)}")

            if parameters:
                for key, value in parameters.items():
                    self.logger.info("The p-value of {} is {}".format(key, value))
                ref_id = parameters.get("ref_id")
                ref_type = parameters.get("ref_type")
            else:
                self.logger.warning(
                    f"REF_ID and REF_TYPE are not present in parameters"
                )
                ref_id = hash_frames_fast(frames)
                ref_type = "checksum_frames"

            results = []
            for i, frame in enumerate(frames):
                try:
                    doc_id = f"doc_{i}"
                    real, mask, blended = self.overlay_processor.segment_frame(
                        doc_id, frame
                    )

                    def _tags(index: int, ftype: str, checksum: str):
                        return {
                            "index": index,
                            "type": ftype,
                            "ttl": 48 * 60,
                            "checksum": checksum,
                        }

                    if self.storage_enabled:
                        frame_checksum = hash_frames_fast(frames=[frame])
                        docs = DocumentArray(
                            [
                                Document(
                                    blob=convert_to_bytes(real),
                                    tags=_tags(i, "real", frame_checksum),
                                ),
                                Document(
                                    blob=convert_to_bytes(mask),
                                    tags=_tags(i, "mask", frame_checksum),
                                ),
                                Document(
                                    blob=convert_to_bytes(blended),
                                    tags=_tags(i, "blended", frame_checksum),
                                ),
                            ]
                        )

                        self.store(
                            ref_id=ref_id,
                            ref_type=ref_type,
                            store_mode="blob",
                            docs=docs,
                        )

                except Exception as e:
                    self.logger.warning(f"Unable to segment document : {e}")

            return results
        except BaseException as error:
            self.logger.error("Extract error", error)
            if self.show_error:
                return {"error": str(error)}
            else:
                return {"error": "inference exception"}
