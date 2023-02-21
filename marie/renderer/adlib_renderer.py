import os
from datetime import datetime
from os import PathLike
from typing import Dict, Any, Union, Optional, Callable

import numpy as np

from marie.logging.predefined import default_logger
from marie.renderer.renderer import ResultRenderer

logger = default_logger


class AdlibRenderer(ResultRenderer):
    def __init__(
        self,
        summary_filename="summary.xml",
        config=None,
    ):
        super().__init__(config)
        if config is None:
            config = {}
        print(f"AdlibRenderer base : {config}")
        self.summary_filename = summary_filename

    @property
    def name(self):
        return "AdlibRenderer"

    def write_adlib_summary_tree(
        self, frames, filename_generator: Callable[[int], str]
    ):
        import xml.etree.ElementTree as gfg

        def _meta(field, val):
            meta = gfg.Element("METADATAELEMENT")
            meta.set("FIELD", str(field))
            meta.set("VALUE", str(val))
            return meta

        root = gfg.Element("OCR")
        metas = gfg.Element("METADATAELEMENTS")

        metas.append(_meta("OCR", "MARIE-AI"))
        metas.append(
            _meta("CreationDate", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        # metas.append(_meta("Title", file_id))

        root.append(metas)
        pages_node = gfg.Element("PAGES")

        for page_index, _path in enumerate(frames):
            pagenumber = page_index + 1
            filename = filename_generator(pagenumber)
            node = gfg.Element("PAGE")
            node.set("Filename", filename)
            node.set("NUMBER", str(pagenumber))
            pages_node.append(node)
        root.append(pages_node)

        tree = gfg.ElementTree(root)
        return tree

    def __render_page(self, image: np.ndarray, result: Dict[str, Any], page_index: int):
        import xml.etree.ElementTree as gfg

        try:
            meta = result["meta"]
            words = result["words"]
            lines = result["lines"]

            # default DPI
            dpi_x = 300.0
            dpi_y = 300.0
            pagenumber = page_index + 1  # TODO : Verify 0 based index

            im_h = meta["imageSize"]["height"] / dpi_y
            im_w = meta["imageSize"]["width"] / dpi_x

            root = gfg.Element("PAGE")
            root.set("HEIGHT", str(im_h))
            root.set("WIDTH", str(im_w))
            root.set("ImageType", "Unknown")
            root.set("NUMBER", str(pagenumber))
            root.set("OCREndTime", "0")
            root.set("OCRStartTime", "0")
            root.set("Producer", "marie")
            root.set("XRESOLUTION", str(dpi_x))
            root.set("YRESOLUTION", str(dpi_y))

            # Add dummy TEXT element
            root.append(gfg.Element("TEXT"))

            for idx, word in enumerate(words):
                x1, y1, w1, h1 = word["box"]
                txt = word["text"]
                x = x1 / dpi_x
                y = y1 / dpi_y
                w = w1 / dpi_x
                h = h1 / dpi_y
                left = x
                right = x + w
                top = y - h
                bottom = y + h
                consecutive = False

                m1 = gfg.Element("TEXTSTRING")
                m1.set("CONSECUTIVE", "FALSE")
                m1.set("FONTNAME", "Courier")
                m1.set("FONTSIZE", "32")
                m1.set("NoLocation", "FALSE")
                m1.set("PageNumber", str(pagenumber))

                m1.set("LEFT", f"{left:.4f}")
                m1.set("RIGHT", f"{right:.4f}")
                m1.set("TOP", f"{top:.4f}")
                m1.set("BOTTOM", f"{bottom:.4f}")
                m1.set("WORD", str(txt))

                root.append(m1)

            tree = gfg.ElementTree(root)
            return tree

        except Exception as ident:
            raise ident

    def render(
        self,
        frames: np.ndarray,
        results: [Dict[str, Any]],
        output_file_or_dir: Union[str, PathLike],
        filename_generator: Optional[Callable[[int], str]] = None,
        **kwargs: Any,
    ) -> None:
        """Renders results into Adlib compatible assets
        Results parameter "format" is expected to be in "XYWH" conversion will be performed to accommodate this
        """

        if not os.path.isdir(output_file_or_dir):
            raise ValueError("output_file_or_dir should be a directory")
        self.logger.info(f"Render Adlib to : {output_file_or_dir}")

        filename_generator = filename_generator or (lambda x: f"{x}.tif.xml")

        for page_index, (image, result) in enumerate(zip(frames, results)):
            try:
                tree = self.__render_page(image, result, page_index)
                output_filename = os.path.join(
                    output_file_or_dir, filename_generator(page_index + 1)
                )
                with open(output_filename, "wb") as fs:
                    tree.write(fs)
            except Exception as e:
                logger.error(e, stack_info=True, exc_info=True)

        tree = self.write_adlib_summary_tree(frames, filename_generator)
        adlib_summary_filename = os.path.join(output_file_or_dir, self.summary_filename)
        with open(adlib_summary_filename, "wb") as ws:
            tree.write(ws)
