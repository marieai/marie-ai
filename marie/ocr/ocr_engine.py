import os
from abc import ABC, abstractmethod
from typing import Union, List, Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from marie.boxes import PSMode, BoxProcessorUlimDit, BoxProcessorCraft
from marie.boxes.box_processor import BoxProcessor
from marie.constants import __model_path__
from marie.document.ocr_processor import OcrProcessor
from marie.logging.logger import MarieLogger
from marie.ocr.coordinate_format import CoordinateFormat
from marie.utils.base64 import encodeToBase64
from marie.utils.image_utils import crop_to_content, hash_frames_fast
from marie.utils.utils import ensure_exists


class OcrEngine(ABC):
    """
    Recognizes text in an image.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        *,
        box_processor: Optional[BoxProcessor] = None,
        **kwargs,
    ) -> None:
        self.logger = MarieLogger(context=self.__class__.__name__)
        work_dir_boxes = ensure_exists("/tmp/boxes")
        self.work_dir_icr = ensure_exists("/tmp/icr")
        ensure_exists("/tmp/fragments")

        # sometimes we have CUDA/GPU support but want to only use CPU
        has_cuda = cuda
        if os.environ.get("MARIE_DISABLE_CUDA"):
            has_cuda = False

        self.has_cuda = has_cuda
        if box_processor is not None:
            self.box_processor = box_processor
        else:
            box_segmentation_mode = int(kwargs.get("box_segmentation_mode", "1"))
            if box_segmentation_mode == 1:
                self.box_processor = BoxProcessorUlimDit(
                    work_dir=work_dir_boxes,
                    cuda=has_cuda,
                )
            elif box_segmentation_mode == 2:
                self.box_processor = BoxProcessorCraft(
                    work_dir=work_dir_boxes, cuda=has_cuda
                )
            else:
                raise Exception(
                    f"Unsupported box segmentation mode : {box_segmentation_mode}"
                )

    @abstractmethod
    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYXY,
        regions: [] = None,
        queue_id: str = None,
        **kwargs,
    ):
        """
        Extract text from the supplied image frames.
        :param frames:
        :param pms_mode:
        :param coordinate_format:
        :param regions:
        :param queue_id:
        :param kwargs:
        """
        ...

    def process_single(
        self,
        box_processor: BoxProcessor,
        icr_processor: OcrProcessor,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        queue_id: str = None,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Process results from OCR engine
        :param box_processor:
        :param icr_processor:
        :param frames:
        :param pms_mode:
        :param coordinate_format:
        :param regions:
        :param queue_id:
        :param kwargs:
        :return:
        """
        queue_id = "0000-0000-0000-0000" if queue_id is None else queue_id
        regions = [] if regions is None else regions
        ro_frames = OcrEngine.copy_frames(frames)
        checksum = hash_frames_fast(ro_frames)
        self.logger.debug(
            "frames , regions , output_format, pms_mode, coordinate_format,"
            f" checksum:  {len(ro_frames)}, {len(regions)}, {pms_mode},"
            f" {coordinate_format}, {checksum}"
        )

        try:
            if len(regions) == 0:
                results = self.__process_extract_fullpage(
                    ro_frames,
                    queue_id,
                    checksum,
                    pms_mode,
                    coordinate_format,
                    box_processor,
                    icr_processor,
                    **kwargs,
                )
            else:
                results = self.__process_extract_regions(
                    ro_frames,
                    queue_id,
                    checksum,
                    pms_mode,
                    regions,
                    box_processor,
                    icr_processor,
                    **kwargs,
                )
            return results
        except BaseException as error:
            self.logger.error("Extract error", exc_info=True)
            raise error

    def __process_extract_fullpage(
        self,
        frames: List[np.ndarray],
        queue_id: str,
        checksum: str,
        pms_mode: PSMode,
        coordinate_format: CoordinateFormat,
        box_processor: BoxProcessor,
        icr_processor: OcrProcessor,
        **kwargs,
    ):
        """Process full page extraction"""
        # Extract each page and augment it with a page in range 1..N+1
        results = []
        # This should be requested as it might not always be desirable to perform this transform
        is_crop_to_content_enabled = kwargs.get("crop_to_content", False)
        padding = 0

        for i, img in enumerate(frames):
            try:
                if is_crop_to_content_enabled:
                    img = crop_to_content(img)
                    padding = 4

                h = img.shape[0]
                w = img.shape[1]

                overlay = (
                    np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
                )
                overlay[padding : h + padding, padding : w + padding] = img

                (
                    boxes,
                    img_fragments,
                    lines,
                    _,
                    line_bboxes,
                ) = box_processor.extract_bounding_boxes(
                    queue_id, checksum, overlay, pms_mode
                )

                result, overlay_image = icr_processor.recognize(
                    queue_id, checksum, overlay, boxes, img_fragments, lines
                )
                # change from xywh -> xyxy
                if CoordinateFormat.XYXY == coordinate_format:
                    self.logger.debug("Changing coordinate format from xywh -> xyxy")
                    for word in result["words"]:
                        x, y, w, h = word["box"]
                        w_box = [x, y, x + w, y + h]
                        word["box"] = w_box
                        # FIXME:  BLOWS memory on GPU
                        # word["box"] = CoordinateFormat.convert(
                        #     word["box"], CoordinateFormat.XYWH, CoordinateFormat.XYXY
                        # )

                # result["overlay_b64"] = encodeToBase64(overlay_image)
                result["meta"]["page"] = i
                result["meta"]["lines"] = lines
                result["meta"]["lines_bboxes"] = line_bboxes
                result["meta"]["format"] = coordinate_format.name.lower()

                results.append(result)
            except Exception as ex:
                self.logger.error(ex)
                raise ex
        return results

    def __process_extract_regions(
        self,
        frames,
        queue_id,
        checksum,
        pms_mode,
        regions,
        box_processor: BoxProcessor,
        icr_processor: OcrProcessor,
        **kwargs,
    ):
        """Process region based extract"""
        filter_snippets = True
        # filter_snippets = (
        #     bool(strtobool(kwargs["filter_snippets"]))
        #     if "filter_snippets" in kwargs
        #     else False
        # )

        # This should be requested as it might not always be desirable to perform this transform
        # crop_to_content_enabled = bool(strtobool(kwargs.get('crop_to_content', False)))
        crop_to_content_enabled = False

        output = []
        extended = []

        for region in regions:
            # validate required fields
            if not all(
                key in region for key in ("id", "pageIndex", "x", "y", "w", "h")
            ):
                raise Exception(f"Required key missing in region : {region}")

            # Additional fields are allowed (e.g. mode)

        # TODO : Introduce mini-batched by region to improve inference
        for region in regions:
            try:
                self.logger.debug(f"Extracting box : {region}")
                rid = region["id"]
                page_index = region["pageIndex"]
                x = region["x"]
                y = region["y"]
                w = region["w"]
                h = region["h"]

                img = frames[page_index]

                if w == 0 or h == 0:
                    self.logger.warning(f"Region has zero width or height : {region}")
                    output.append({"id": rid, "text": "", "confidence": 0.0})
                    continue

                if y + h > img.shape[0] or x + w > img.shape[1]:
                    self.logger.warning(f"Region out of bounds : {region}")
                    output.append({"id": rid, "text": "", "confidence": 0.0})
                    continue

                img = img[y : y + h, x : x + w].copy()
                # allow for small padding around the component
                padding = 4
                if crop_to_content_enabled:
                    img = crop_to_content(img)
                    h = img.shape[0]
                    w = img.shape[1]
                    padding = 4

                if padding != 0:
                    overlay = (
                        np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8)
                        * 255
                    )
                    overlay[padding : h + padding, padding : w + padding] = img
                else:
                    overlay = img

                # cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay)
                # each region can have its own segmentation mode
                if "mode" in region:
                    mode = PSMode.from_value(region["mode"])
                else:
                    mode = pms_mode

                (
                    boxes,
                    img_fragments,
                    lines,
                    _,
                    lines_bboxes,
                ) = box_processor.extract_bounding_boxes(
                    queue_id, checksum, overlay, psm=mode
                )

                result, overlay_image = icr_processor.recognize(
                    queue_id, checksum, overlay, boxes, img_fragments, lines
                )

                del boxes
                del img_fragments
                del lines
                del lines_bboxes

                if not filter_snippets:
                    result["overlay_b64"] = encodeToBase64(overlay_image)

                result["id"] = rid
                extended.append(result)

                # TODO : Implement rendering modes
                # 1 - Simple
                # 2 - Full
                # 3 - HOCR
                self.logger.debug(result)
                rendering_mode = "simple"
                region_result = {}
                if rendering_mode == "simple":
                    if "lines" in result and len(result["lines"]) > 0:
                        lines = result["lines"]
                        line = lines[0]
                        region_result["id"] = rid
                        region_result["text"] = line["text"]
                        region_result["confidence"] = line["confidence"]
                        output.append(region_result)
                    else:
                        output.append({"id": rid, "text": "", "confidence": 0.0})

            except Exception as ex:
                self.logger.error(ex)
                raise ex

        # Filter out base 64 encoded fragments(fragment_b64, overlay_b64)
        # This is useful when we like to display or process image in the output but has significant payload overhead

        def filter_base64(node, filters):
            if isinstance(node, (list, tuple, np.ndarray)):
                for v in node:
                    filter_base64(v, filters)
            elif isinstance(node, dict):
                for flt in filters:
                    try:
                        del node[flt]
                    except KeyError:
                        pass
                for key, value in node.items():
                    filter_base64(value, filters)
            else:
                pass
            return node

        if filter_snippets:
            extended = filter_base64(extended, filters=["fragment_b64", "overlay_b64"])

        return {"regions": output, "extended": extended}

    @staticmethod
    def copy_frames(
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
    ) -> List[np.ndarray]:
        """
        Copies the frames (deep copy) to a new list of frames and returns it .
        :param frames:  list of frames
        :return:  list of frames
        """
        ro_frames = []  # [None] * len(frames)
        # we don't want to modify the original Numpy/PIL image as the caller might be depended on the original type
        for idx, frame in enumerate(frames):
            f = frame
            if isinstance(frame, Image.Image):
                converted = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                f = converted
            ro_frames.append(f.copy())
        return ro_frames
