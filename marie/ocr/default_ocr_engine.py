import os
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode, BoxProcessorCraft
from marie.constants import __model_path__
from marie.document import TrOcrIcrProcessor
from marie.logging.logger import MarieLogger
from marie.ocr import OcrEngine, CoordinateFormat
from marie.utils.image_utils import hash_frames_fast
from marie.utils.utils import ensure_exists


class DefaultOcrEngine(OcrEngine):
    """
    Recognizes text in an image.
    This implementation will select best available OcrEngine based on available models and configs
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(models_dir=models_dir, cuda=cuda, **kwargs)
        self.icr_processor = TrOcrIcrProcessor(
            work_dir=self.work_dir_icr, cuda=self.has_cuda
        )

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        queue_id: str = None,
        **kwargs: Any,
    ) -> List[Dict]:
        try:
            queue_id = "0000-0000-0000-0000" if queue_id is None else queue_id
            regions = [] if regions is None else regions

            ro_frames = []  # [None] * len(frames)
            # we don't want to modify the original Numpy/PIL image as the caller might be depended on the original type
            for idx, frame in enumerate(frames):
                f = frame
                if isinstance(frame, Image.Image):
                    converted = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    f = converted
                ro_frames.append(f.copy())

            # calculate hash based on the image frame
            checksum = hash_frames_fast(ro_frames)

            self.logger.debug(
                "frames , regions , output_format, pms_mode, coordinate_format,"
                f" checksum:  {len(ro_frames)}, {len(regions)}, {pms_mode},"
                f" {coordinate_format}, {checksum}"
            )

            if len(regions) == 0:
                results = self.__process_extract_fullpage(
                    ro_frames,
                    queue_id,
                    checksum,
                    pms_mode,
                    coordinate_format,
                    self.box_processor,
                    self.icr_processor,
                    **kwargs,
                )
            else:
                results = self.__process_extract_regions(
                    ro_frames,
                    queue_id,
                    checksum,
                    pms_mode,
                    regions,
                    self.box_processor,
                    self.icr_processor,
                    **kwargs,
                )

            return results
        except BaseException as error:
            self.logger.error("Extract error", exc_info=True)
            raise error
