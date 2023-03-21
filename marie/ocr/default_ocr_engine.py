import copy
import os
from distutils.util import strtobool as strtobool
from typing import Any, Dict, List, Union
import cv2
import numpy as np
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode, BoxProcessorCraft
from marie.constants import __model_path__
from marie.document import TrOcrIcrProcessor
from marie.logging.logger import MarieLogger
from marie.ocr import OcrEngine, CoordinateFormat
from marie.utils.base64 import encodeToBase64
from marie.utils.image_utils import hash_frames_fast, crop_to_content
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
        super().__init__(**kwargs)
        self.logger = MarieLogger(context=self.__class__.__name__)
        work_dir_boxes = ensure_exists("/tmp/boxes")
        work_dir_icr = ensure_exists("/tmp/icr")
        ensure_exists("/tmp/fragments")

        # sometimes we have CUDA/GPU support but want to only use CPU
        has_cuda = cuda
        if os.environ.get("MARIE_DISABLE_CUDA"):
            has_cuda = False

        box_segmentation_mode = int(kwargs.get("box_segmentation_mode", "1"))
        box_segmentation_mode = 1
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

        self.icr_processor = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=has_cuda)

    def __process_extract_fullpage(
        self,
        frames: np.ndarray,
        queue_id: str,
        checksum: str,
        pms_mode: PSMode,
        coordinate_format: CoordinateFormat,
        **kwargs,
    ):
        """Process full page extraction"""
        # Extract each page and augment it with a page in range 1..N+1
        results = []
        # This should be requested as it might not always be desirable to perform this transform
        is_crop_to_content_enabled = kwargs.get('crop_to_content', False)
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
                ) = self.box_processor.extract_bounding_boxes(
                    queue_id, checksum, overlay, pms_mode
                )

                result, overlay_image = self.icr_processor.recognize(
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
        self, frames, queue_id, checksum, pms_mode, regions, **kwargs
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
                img = img[y : y + h, x : x + w].copy()
                # allow for small padding around the component
                padding = 0
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

                cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay)
                (
                    boxes,
                    img_fragments,
                    lines,
                    _,
                    lines_bboxes,
                ) = self.box_processor.extract_bounding_boxes(
                    queue_id, checksum, overlay, pms_mode
                )

                result, overlay_image = self.icr_processor.recognize(
                    queue_id, checksum, overlay, boxes, img_fragments, lines
                )

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
                    if "lines" in result:
                        lines = result["lines"]
                        line = lines[0]
                        region_result["id"] = rid
                        region_result["text"] = line["text"]
                        region_result["confidence"] = line["confidence"]
                        output.append(region_result)
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

    def extract(
        self,
        frames: Union[np.ndarray, List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        queue_id: str = None,
        **kwargs: Any,
    ):
        try:
            queue_id = "0000-0000-0000-0000" if queue_id is None else queue_id
            regions = [] if regions is None else regions
            ro_frames = []
            # we don't want to modify the original Numpy/PIL image as the caller might be depended on the original type
            for _, frame in enumerate(frames):
                if isinstance(frame, Image.Image):
                    converted = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    f = copy.deepcopy(converted)
                else:
                    f = copy.deepcopy(frame)
                ro_frames.append(f)

            # calculate hash based on the image frame
            # ro_frames = np.array(ro_frames)
            checksum = hash_frames_fast(ro_frames)

            self.logger.debug(
                "frames , regions , output_format, pms_mode, coordinate_format,"
                f" checksum:  {len(ro_frames)}, {len(regions)}, {pms_mode},"
                f" {coordinate_format}, {checksum}"
            )

            if len(regions) == 0:
                results = self.__process_extract_fullpage(
                    ro_frames, queue_id, checksum, pms_mode, coordinate_format, **kwargs
                )
            else:
                results = self.__process_extract_regions(
                    ro_frames, queue_id, checksum, pms_mode, regions, **kwargs
                )

            # store_json_object(results, '/tmp/fragments/results-complex.json')
            return results
        except BaseException as error:
            self.logger.error("Extract error", error)
            raise error
