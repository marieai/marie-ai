import logging
import re
from typing import Optional

from marie import check
from marie.extract.schema import Segment
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.extract.structures.line_with_meta import LineWithMeta


def locate_line(
    lines_for_page: list[LineWithMeta], line_number: int
) -> Optional[LineWithMeta]:
    """
    Retrieve a specific line by page_id and line_number.
    """
    check.int_param(line_number, "line_number")
    for line in lines_for_page:
        if line.metadata.line_id == line_number:
            return line
    return None


def extract_page_id(filename: str) -> int:
    """
    Extracts the page id from a filename.
    """
    # match = re.match(r".*_?(\d+)\.(tif|png)\.json", filename)
    match = re.match(r"(?:.*_)?(\d+)\.(tif|png)\.json", filename)
    if not match:
        logging.warning(f"Skipping file with unexpected format: {filename}")
        return -1
    return int(match.group(1))


def _annotate_segment(
    annotation_type: str, line: LineWithMeta, segment: Segment
) -> None:
    """
    Annotates a line with metadata extracted from a segment.

    The function updates the provided line object by adding a SegmentAnnotation
    based on the extracted segment details. This operation assumes the
    line already contains existing annotations; otherwise, an error will
    be raised.

    Raises:
    ValueError: If the provided line does not contain any existing
    annotations.
    """

    if segment is None:
        return
    if line.annotations is None:
        raise ValueError(f"{line} does not contain any annotations")

    # TODO : Need to implement locating the string in the Line segment
    name = segment.label
    value = segment.value
    annotation = TypedAnnotation(1, 1, name, value, annotation_type, [])
    line.annotations.append(annotation)
