import json
import os
from datetime import datetime
from distutils.util import strtobool as strtobool
from enum import Enum
from typing import Dict, Union, Optional, TYPE_CHECKING

import numpy as np
import torch
from docarray import DocumentArray
from rich import print
from torch.backends import cudnn

from marie import Executor, requests
from marie.api import value_from_payload_or_args

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.document import TrOcrIcrProcessor
from marie.numpyencoder import NumpyEncoder
from marie.renderer.text_renderer import TextRenderer
from marie.utils.base64 import encodeToBase64
from marie.utils.docs import array_from_docs
from marie.utils.image_utils import hash_bytes
from marie.utils.utils import ensure_exists
from marie.logging.predefined import default_logger

logger = default_logger


class OutputFormat(Enum):
    """Output format for the document"""

    JSON = "json"  # Render document as JSON output
    PDF = "pdf"  # Render document as PDF
    TEXT = "text"  # Render document as plain TEXT
    ASSETS = "assets"  # Render and return all available assets

    @staticmethod
    def from_value(value: str):
        if value is None:
            return OutputFormat.JSON
        for data in OutputFormat:
            if data.value == value.lower():
                return data
        return OutputFormat.JSON


class CoordinateFormat(Enum):
    """Output format for the words
    defaults to : xywh
    """

    XYWH = "xywh"  # Default
    XYXY = "xyxy"

    @staticmethod
    def from_value(value: str):
        if value is None:
            return CoordinateFormat.XYWH
        for data in CoordinateFormat:
            if data.value == value.lower():
                return data
        return CoordinateFormat.XYWH

    @staticmethod
    def convert(
        box: np.ndarray, from_mode: "CoordinateFormat", to_mode: "CoordinateFormat"
    ) -> np.ndarray:
        """
        Args:
            box: can be a 4-tuple,
            from_mode, to_mode (CoordinateFormat)

        Ref : Detectron boxes
        Returns:
            The converted box of the same type.
        """
        arr = np.array(box)
        assert arr.shape == (4,), "CoordinateFormat.convert takes either a 4-tuple/list"

        if from_mode == to_mode:
            return box

        original_type = type(box)
        original_shape = arr.shape
        arr = arr.reshape(-1, 4)

        if to_mode == CoordinateFormat.XYXY and from_mode == CoordinateFormat.XYWH:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == CoordinateFormat.XYXY and to_mode == CoordinateFormat.XYWH:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise RuntimeError("Cannot be here!")

        return original_type(arr.flatten())


class TextExtractionExecutor(Executor):
    """
    Executor for extracting text.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors

        work_dir_boxes = ensure_exists("/tmp/boxes")
        work_dir_icr = ensure_exists("/tmp/icr")

        # sometimes we have CUDA/GPU support but want to only use CPU
        has_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            has_cuda = False

        if has_cuda:
            # benchmark mode is good whenever your input sizes for your network do not vary
            cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True

        # self.box_processor = BoxProcessorCraft(work_dir=work_dir_boxes, cuda=has_cuda)

        if True:
            self.box_processor = BoxProcessorUlimDit(
                work_dir=work_dir_boxes,
                models_dir="../model_zoo/unilm/dit/text_detection",
                cuda=True,
            )
        self.icr_processor = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=has_cuda)

    @requests(on="/text/status")
    def info(self, **kwargs):
        logger.info(f"Self : {self}")
        return {"index": "complete"}

    def __process_extract_fullpage(
        self,
        frames: np.ndarray,
        queue_id: str,
        checksum: str,
        pms_mode: PSMode,
        coordinate_format: CoordinateFormat,
        **kwargs,
    ):
        """
        Process full page extraction
        """
        # Extract each page and augment it with a page in range 1..N+1
        results = []
        assets = []

        for i, img in enumerate(frames):
            h = img.shape[0]
            w = img.shape[1]
            # allow for small padding around the component
            padding = 0
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
                logger.info("Changing coordinate format from xywh -> xyxy")
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

        return results

    def __process_extract_regions(
        self, frames, queue_id, checksum, pms_mode, regions, **kwargs
    ):
        """Process region based extract"""
        filter_snippets = (
            bool(strtobool(kwargs["filter_snippets"]))
            if "filter_snippets" in kwargs
            else False
        )
        output = []
        extended = []

        for region in regions:
            # validate required fields
            if not all(
                key in region for key in ("id", "pageIndex", "x", "y", "w", "h")
            ):
                raise Exception(f"Required key missing in region : {region}")

        # allow for small padding around the component
        padding = 0

        for region in regions:
            try:
                logger.info(f"Extracting box : {region}")
                rid = region["id"]
                page_index = region["pageIndex"]
                x = region["x"]
                y = region["y"]
                w = region["w"]
                h = region["h"]

                img = frames[page_index]
                img = img[y : y + h, x : x + w].copy()
                overlay = img

                if padding != 0:
                    overlay = (
                        np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8)
                        * 255
                    )
                    overlay[padding : h + padding, padding : w + padding] = img

                # cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay)
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

                logger.info(result)
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
                logger.error(ex)
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

    @requests(on="/text/status")
    def status(self, parameters, **kwargs):
        logger.info(f"Self : {self}")
        return {"index": "complete"}

    @requests(on="/text/extract")
    def extract(self, parameters, docs: Optional[DocumentArray] = None, **kwargs):
        """Load the image from `uri`, extract text and bounding boxes.
        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        logger.info("Starting ICR processing request")

        for doc in docs:
            print(doc.tensor)

        queue_id: str = parameters.get("queue_id", "0000-0000-0000-0000")
        for key, value in parameters.items():
            logger.info("The value of {} is {}".format(key, value))

        try:
            if "payload" not in parameters or parameters["payload"] is None:
                return {"error": "empty payload"}
            else:
                payload = parameters["payload"]

            regions = payload["regions"] if "regions" in payload else []

            # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
            coordinate_format = CoordinateFormat.from_value(
                value_from_payload_or_args(payload, "format", default="xywh")
            )
            pms_mode = PSMode.from_value(
                value_from_payload_or_args(payload, "mode", default="")
            )
            output_format = OutputFormat.from_value(
                value_from_payload_or_args(payload, "output", default="json")
            )

            frames = array_from_docs(docs)
            frame_len = len(frames)
            # convert frames into a checksum
            src = []
            for i, frame in enumerate(frames):
                src = np.append(src, np.ravel(frame))
            checksum = hash_bytes(src)

            logger.info(
                "frames , regions , output_format, pms_mode, coordinate_format,"
                f" checksum:  {frame_len}, {len(regions)}, {output_format}, {pms_mode},"
                f" {coordinate_format}, {checksum}"
            )

            if len(regions) == 0:
                results = self.__process_extract_fullpage(
                    frames, queue_id, checksum, pms_mode, coordinate_format
                )
            else:
                results = self.__process_extract_regions(
                    frames, queue_id, checksum, pms_mode, regions
                )

            output = None

            if output_format == OutputFormat.JSON:
                output = self.render_as_json(queue_id, checksum, frames, results)
            elif output_format == OutputFormat.PDF:
                # renderer = PdfRenderer(config={"preserve_interword_spaces": True})
                # renderer.render(image, result, output_filename)
                raise Exception("PDF Not implemented")
            elif output_format == OutputFormat.TEXT:
                output = self.render_as_text(queue_id, checksum, frames, results)
            elif output_format == OutputFormat.ASSETS:
                output = self.render_as_assets(queue_id, checksum, frames, results)

            return results
        except BaseException as error:
            logger.error("Extract error", error)
            if self.show_error:
                return {"error": str(error)}
            else:
                return {"error": "inference exception"}

    def render_as_json(self, queue_id, checksum, frames, results) -> Dict:
        """Renders specific results as JSON"""
        if False:
            output = json.dumps(
                results,
                sort_keys=False,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=2,
                cls=NumpyEncoder,
            )

        return results

    def render_as_text(self, queue_id, checksum, frames, results) -> str:
        """Renders specific results as text"""
        try:
            work_dir = ensure_exists(f"/tmp/marie/{queue_id}")
            str_current_datetime = str(datetime.now())
            output_filename = f"{work_dir}/{checksum}_{str_current_datetime}.txt"

            renderer = TextRenderer(config={"preserve_interword_spaces": True})
            output = renderer.render(frames, results, output_filename)
            return output

        except BaseException as e:
            logger.error("Unable to render TEXT for document", e)

    def render_as_assets(self, queue_id, checksum, frames, results):
        """Render all documents as assets"""

        json_results = self.render_as_json(queue_id, checksum, frames, results)
        text_results = self.render_as_text(queue_id, checksum, frames, results)

        raise Exception("Not Implemented")


class ExtractExecutor(Executor):
    def __init__(
        self,
        name: str = '',
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        """
        :param device: 'cpu' or 'cuda'. Default is None, which auto-detects the device.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param minibatch_size: The size of the minibatch for preprocessing and encoding. Default is 32. Reduce this
            number if you encounter OOM errors.
        :param dtype: inference data type, if None defaults to torch.float32 if device == 'cpu' else torch.float16.
        """
        super().__init__(**kwargs)

    @requests(on="/text/extract")
    def extract(self, parameters, docs: Optional[DocumentArray] = None, **kwargs):
        default_logger.info(f"Executing extract : {len(docs)}")
        default_logger.info(kwargs)
        default_logger.info(parameters)

        logger.info("Processing docs : ")
        logger.info(docs)
        import threading
        import time

        time.sleep(1.3)

        for doc in docs:
            doc.text = f"{doc.text} : >> {threading.current_thread().name}  : {threading.get_ident()}"

    @requests(on="/status")
    def status(self, parameters, **kwargs):
        logger.info(f"Self : {self}")
        return {"index": "complete"}
