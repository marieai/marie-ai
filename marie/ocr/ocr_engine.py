import os
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from marie.boxes import BoxProcessorCraft, BoxProcessorUlimDit, PSMode
from marie.boxes.box_processor import BoxProcessor
from marie.constants import __model_path__
from marie.document.ocr_processor import OcrProcessor
from marie.logging.logger import MarieLogger
from marie.ocr.coordinate_format import CoordinateFormat
from marie.utils.base64 import encodeToBase64
from marie.utils.image_utils import crop_to_content, hash_frames_fast
from marie.utils.utils import ensure_exists

bbox_cache = {}


def reset_bbox_cache():
    global bbox_cache
    bbox_cache = {}


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
        bbox_results_batch = []
        print(f'regions: {len(regions)}')

        pages = {}
        for region in regions:
            pages.setdefault(region["pageIndex"], []).append(region)

        # Batch region by page
        for page_index, regions in pages.items():
            # TODO : Introduce mini-batched by region to improve inference
            img = frames[page_index]
            x_batch, y_batch, w_batch, h_batch = img.shape[1], img.shape[0], 0, 0
            region_ids = []
            for region in regions:
                try:
                    self.logger.debug(f"Extracting box : {region}")
                    rid = region["id"]
                    region_ids.append(rid)
                    x = region["x"]
                    y = region["y"]
                    w = region["w"]
                    h = region["h"]

                    if w == 0 or h == 0:
                        self.logger.warning(
                            f"Region has zero width or height : {region}"
                        )
                        output.append({"id": rid, "text": "", "confidence": 0.0})
                        continue

                    if y + h > img.shape[0] or x + w > img.shape[1]:
                        self.logger.warning(f"Region out of bounds : {region}")
                        output.append({"id": rid, "text": "", "confidence": 0.0})
                        continue

                    # Update the size of the batch overlay
                    x_batch = min(x, x_batch)
                    y_batch = min(y, y_batch)
                    w_batch = max(x + w, x_batch + w_batch) - x_batch
                    h_batch = max(y + h, h_batch + y_batch) - y_batch

                    region_fragment = img[y : y + h, x : x + w].copy()
                    # allow for small padding around the component
                    padding = 4
                    if crop_to_content_enabled:
                        region_fragment = crop_to_content(region_fragment)
                        h = region_fragment.shape[0]
                        w = region_fragment.shape[1]
                        padding = 4

                    if padding != 0:
                        region_overlay = (
                            np.ones(
                                (h + padding * 2, w + padding * 2, 3), dtype=np.uint8
                            )
                            * 255
                        )
                        region_overlay[
                            padding : h + padding, padding : w + padding
                        ] = region_fragment
                    else:
                        region_overlay = region_fragment

                    # cv2.imwrite(f"/tmp/marie/region_overlay_{page_index}_{rid}.png", region_overlay)
                    # each region can have its own segmentation mode
                    if "mode" in region:
                        mode = PSMode.from_value(region["mode"])
                    else:
                        mode = pms_mode

                    # create cache key from region id and overlay hash
                    region_overlay_hash = hash_frames_fast(region_overlay)
                    cache_key = f"{id(region)}_{region_overlay_hash}"
                    bbox_results = None

                    if cache_key in bbox_cache:
                        bbox_results = bbox_cache[cache_key]

                    if bbox_results is None:
                        bbox_results = box_processor.extract_bounding_boxes(
                            queue_id, checksum, region_overlay, psm=mode
                        )
                        bbox_cache[cache_key] = bbox_results

                    bbox_results_batch.append(bbox_results)

                except Exception as ex:
                    self.logger.error(ex)
                    raise ex

            # use a crop of the image related to the batch
            batch_crop = img[y_batch : y_batch + h_batch, x_batch : x_batch + w_batch]
            (boxes, img_fragments, lines, _, lines_bboxes,) = (
                list(chain.from_iterable(x))
                for i, x in enumerate(zip(*bbox_results_batch))
            )
            batch_result, batch_overlay_image = icr_processor.recognize(
                queue_id, checksum, batch_crop, boxes, img_fragments, lines
            )

            del boxes
            del img_fragments
            del lines
            del lines_bboxes

            if not filter_snippets:
                batch_result["overlay_b64"] = encodeToBase64(batch_overlay_image)

            extended.append(batch_result)

            # TODO : Implement rendering modes
            # 1 - Simple
            # 2 - Full
            # 3 - HOCR
            rendering_mode = "simple"
            if rendering_mode == "simple":
                # unpack result from batch icr
                if "words" in batch_result and len(batch_result["words"]) == len(
                    region_ids
                ):
                    for words, rid in zip(batch_result["words"], region_ids):
                        region_result = {
                            "id": rid,
                            "text": words["text"],
                            "confidence": words["confidence"],
                        }
                        output.append(region_result)
                else:
                    for rid in region_ids:
                        output.append({"id": rid, "text": "", "confidence": 0.0})
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
