from datetime import datetime
from os import PathLike
from typing import Dict, Any, Union

import numpy as np

from marie.ocr import OutputFormat
from marie.logging.logger import MarieLogger
from marie.renderer import TextRenderer

from marie.utils import json
from marie.utils.json import to_json
from marie.utils.utils import ensure_exists

from marie.logging.predefined import default_logger


class TextOutputRenderer:
    def __init__(self, **kwargs):
        self.logger = MarieLogger(self.__class__.__name__)

    def render(
        self,
        output_format: OutputFormat,
        queue_id: str,
        checksum: str,
        frames: [np.array],
        results: [Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Result renderer that renders results to output

        Args:
            output_format (OutputFormat: The format to output data in
            frames ([np.array]):  Frames
            results ([[Dict[str, Any]]): A OCR results array
            output_filename (Union[str, PathLike]): a file path which exists on the local file system
        Returns:
            None
        """

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

        return None

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
            self.logger.error("Unable to render TEXT for document", e)

    def render_as_assets(self, queue_id, checksum, frames, results):
        """Render all documents as assets"""

        json_results = self.render_as_json(queue_id, checksum, frames, results)
        text_results = self.render_as_text(queue_id, checksum, frames, results)

        raise Exception("Not Implemented")

    def render_as_json(self, queue_id, checksum, frames, results) -> str:
        """Renders specific results as JSON"""
        output = to_json(results)

        return output
