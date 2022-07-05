from enum import Enum
from typing import Optional, Dict

import torch
from docarray import DocumentArray
from torch.backends import cudnn

from marie import Executor, requests

import os

import json
import logging
from distutils.util import strtobool as strtobool

import numpy as np

import cv2

from flask import jsonify

from marie.logging.logger import MarieLogger
from marie.renderer import PdfRenderer
from marie.renderer.text_renderer import TextRenderer
from marie.serve.runtimes import monitoring
from marie.utils.docs import array_from_docs
from marie.utils.utils import ensure_exists
from marie.utils.base64 import base64StringToBytes, encodeToBase64
from marie.utils.network import get_ip_address

from marie.boxes import PSMode
from marie.boxes import BoxProcessorCraft
from marie.numpyencoder import NumpyEncoder
from marie.document import TrOcrIcrProcessor
from datetime import datetime

logger = MarieLogger("")


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
            cudnn.benchmark = False
            cudnn.deterministic = False

        self.box_processor = BoxProcessorCraft(work_dir=work_dir_boxes, cuda=has_cuda)
        self.icr_processor = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=has_cuda)

    def status(self, **kwargs):
        """Get application status"""
        import os

        build = {}
        if os.path.exists(".build"):
            try:
                with open(".build", "r") as fp:
                    build = json.load(fp)
            except Exception as ex:
                build = {"status": "Unable to read .build file"}
                logger.error(ex)

        host = get_ip_address()

        return (
            jsonify(
                {
                    "name": "marie-ai",
                    "host": host,
                    "component": [
                        {"name": "craft", "version": "1.0.0"},
                        {"name": "craft-benchmark", "version": "1.0.0"},
                        {"name": "trocr", "version": "1.0.0"},
                    ],
                    "build": build,
                }
            ),
            200,
        )

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
            # change from xywy -> xyxy
            if CoordinateFormat.XYXY == coordinate_format:
                logger.info("Changing coordinate format from xyhw -> xyxy")
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
                # allow for small padding around the component
                padding = 4
                overlay = (
                    np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
                )
                overlay[padding : h + padding, padding : w + padding] = img
                cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay)

                logger.info(f"pms_mode = {pms_mode}")
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

                # cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay_image)
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

    # @requests()
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        Load the image from `uri`, extract text and bounding boxes.
         Args:
             docs : Documents to process
             queue_id: Unique queue to tie the extraction to
        """
        queue_id: str = kwargs.get("queue_id", "0000-0000-0000-0000")
        for key, value in kwargs.items():
            print("The value of {} is {}".format(key, value))

        logger.info("Starting ICR processing request", extra={"session": queue_id})

        try:
            if "payload" not in kwargs or kwargs["payload"] is None:
                return {"error": "empty payload"}, 200
            else:
                payload = kwargs["payload"]

            pms_mode = PSMode.from_value(payload["mode"] if "mode" in payload else "")

            coordinate_format = CoordinateFormat.from_value(
                payload["format"] if "format" in payload else "xywh"
            )
            output_format = OutputFormat.from_value(
                payload["output"] if "output" in payload else "json"
            )

            regions = payload["regions"] if "regions" in payload else []

            # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
            if "args" in payload:
                pms_mode = PSMode.from_value(
                    payload["args"]["mode"] if "mode" in payload["args"] else ""
                )
                output_format = OutputFormat.from_value(
                    payload["args"]["output"] if "output" in payload["args"] else "json"
                )

            frames = array_from_docs(docs)
            checksum = str(abs(hash(frames.data.tobytes())))
            frame_len = len(frames)

            logger.info(
                f"frames , regions , output_format, pms_mode, coordinate_format, checksum:  {frame_len}, {len(regions)}, {output_format}, {pms_mode}, {coordinate_format}, {checksum}"
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

            return output
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
