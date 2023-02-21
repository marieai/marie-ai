import os.path
from os import PathLike
from typing import Dict, Any, Union, Optional, Callable
from xml.sax.saxutils import escape

import numpy as np

from marie.logging.predefined import default_logger
from marie.renderer.renderer import ResultRenderer

logger = default_logger


class BlobRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)

    @property
    def name(self):
        return "BlobRenderer"

    # def render(self, img, result, output_filename):
    def __render_page(self, image: np.ndarray, result: Dict[str, Any], page_index: int):
        import xml.etree.ElementTree as gfg

        # TODO : This needs to be configurable
        root = gfg.Element("blobs")
        root.set("angle", "0.0")
        root.set("yres", "300")
        root.set("xres", "300")
        root.set("page", str(page_index))

        try:
            meta = result["meta"]
            words = result["words"]
            lines = result["lines"]

            # self.page_number
            for idx, word in enumerate(words):
                text = word["text"]
                x, y, w, h = word["box"]
                m1 = gfg.Element("blob")
                m1.set("x", str(x))
                m1.set("y", str(y))
                m1.set("w", str(w))
                m1.set("h", str(h))
                m1.set("text", escape(text))

                b1 = gfg.SubElement(m1, "page")
                b1.text = str(
                    page_index + 1
                )  ## TODO :VERIFY INDEXING > this should be 1 based

                root.append(m1)

            tree = gfg.ElementTree(root)
            return tree
        except Exception as ident:
            raise ident

    def render(
        self,
        frames: np.ndarray,
        results: [Dict[str, Any]],
        output_path: Union[str, PathLike],
        filename_generator: Optional[Callable[[int], str]] = None,
    ) -> None:
        """Renders results into BLOBS
        Results parameter "format" is expected to be in "XYWH" conversion will be performed to accommodate this
        """

        if not os.path.isdir(output_path):
            raise ValueError("output_path should be a directory")

        self.logger.info(f"Render BLOBS to : {output_path}")

        filename_generator = filename_generator or (lambda x: f"{x}.BLOBS.XML")

        # The underlying ByteIO buffer will be closed when we write the file out
        for page_index, (image, result) in enumerate(zip(frames, results)):
            try:
                tree = self.__render_page(image, result, page_index)
                output_filename = os.path.join(
                    output_path, filename_generator(page_index + 1)
                )
                with open(output_filename, "wb") as fs:
                    tree.write(fs)

            except Exception as e:
                logger.error(e, stack_info=True, exc_info=True)
