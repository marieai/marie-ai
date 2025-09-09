import os
from typing import Callable, Optional

import cv2
from PIL import Image, ImageDraw

from marie.executor.ner.utils import draw_box, get_font


class AnnotationHighlighter:
    """
    Generic annotation highlighter for UnstructuredDocument pages.
    - color_by: "type", "name", or "fixed"
    - colors: dict that maps type/name -> (R, G, B, A). If not provided, a default color is used.
    - label_fn: optional callable to render a label for each annotation (annotation, line) -> str | None
    - filter_fn: optional callable to decide whether to draw an annotation (annotation, line) -> bool
    - use_annotation_bboxes: prefer annotation-level bboxes when available, otherwise use line bbox
    - output_name_fn: optional callable to customize output filename per page
    """

    def __init__(
        self,
        color_by: str = "type",
        colors: Optional[dict[str, tuple[int, int, int, int]]] = None,
        default_color: tuple[int, int, int, int] = (255, 0, 0, 128),
        label_fn: Optional[Callable[[object, object], Optional[str]]] = None,
        filter_fn: Optional[Callable[[object, object], bool]] = None,
        use_annotation_bboxes: bool = True,
        font_size: int = 14,
        output_name_fn: Optional[Callable[[int], str]] = None,
    ) -> None:
        self.color_by = color_by  # "type" | "name" | "fixed"
        self.colors = colors or {}
        self.default_color = default_color
        self.label_fn = label_fn
        self.filter_fn = filter_fn
        self.use_annotation_bboxes = use_annotation_bboxes
        self.font_size = font_size
        self.output_name_fn = output_name_fn

    def _color_for(self, annotation) -> tuple[int, int, int, int]:
        if self.color_by == "fixed":
            return self.default_color

        key = None
        try:
            if self.color_by == "type":
                key = annotation.annotation_type
            elif self.color_by == "name":
                key = annotation.name
        except AttributeError:
            key = None

        if key in self.colors:
            return self.colors[key]
        return self.default_color

    def _iter_targets(self, doc, page_id: int):
        lines = doc.lines_for_page(page_id)
        for line in lines:
            anns = None
            try:
                anns = line.annotations
            except AttributeError:
                anns = None
            if not anns:
                continue
            for ann in anns:
                if self.filter_fn is not None:
                    try:
                        if not self.filter_fn(ann, line):
                            continue
                    except Exception:
                        continue
                yield line, ann

    def _draw_bbox(self, draw, bbox, color, font, label: Optional[str]) -> None:
        # bbox is expected as (x, y, w, h)
        draw_box(draw, bbox, label, color, font)

    def _collect_bboxes(self, line, annotation) -> list[tuple[int, int, int, int]]:
        if self.use_annotation_bboxes:
            try:
                bboxes = annotation.bboxes
            except AttributeError:
                bboxes = None
            if isinstance(bboxes, list) and len(bboxes) > 0:
                return [
                    tuple(b)
                    for b in bboxes
                    if isinstance(b, (list, tuple)) and len(b) == 4
                ]

        try:
            bbox = line.metadata.model.bbox
            return [tuple(bbox)]
        except AttributeError:
            return []

    def highlight_document(
        self, doc, frames: list, output_dir: str, output_suffix: str = "ann"
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        font = get_font(self.font_size)

        for page_id in range(doc.page_count):
            frame = frames[page_id]
            if not isinstance(frame, Image.Image):
                _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                viz_img = Image.fromarray(frame)
            else:
                viz_img = frame.copy()

            draw = ImageDraw.Draw(viz_img, "RGBA")

            for line, ann in self._iter_targets(doc, page_id):
                color = self._color_for(ann)
                label = None
                if self.label_fn is not None:
                    try:
                        label = self.label_fn(ann, line)
                    except Exception:
                        label = None
                for bbox in self._collect_bboxes(line, ann):
                    self._draw_bbox(draw, bbox, color, font, label)

            if self.output_name_fn is not None:
                out_name = self.output_name_fn(page_id)
            else:
                out_name = f"{page_id + 1}-{output_suffix}.png"

            viz_img.save(os.path.join(output_dir, out_name))
