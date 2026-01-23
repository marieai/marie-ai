import logging
import os
import re
from collections import OrderedDict
from functools import lru_cache
from itertools import cycle
from typing import Callable, List, Optional, Type, Union

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from pydantic import ValidationError

from marie.components.table_rec import TableRecPredictor
from marie.constants import __model_path__
from marie.executor.ner.utils import draw_box, get_font, visualize_icr
from marie.extract.models.match import SubzeroResult
from marie.extract.readers.meta_reader.meta_reader import MetaReader
from marie.extract.registry import component_registry
from marie.extract.results.base import _annotate_segment, extract_page_id, locate_line
from marie.extract.results.result_converter import convert_document_to_structure
from marie.extract.results.result_validator import (
    validate_document,
    validate_parser_output,
)
from marie.extract.results.table_parsers import _parse_table_mrp, _parse_table_plain
from marie.extract.results.util import generate_distinct_colors
from marie.extract.results.validation_report import generate_validation_report
from marie.extract.schema import ExtractionResult, Segment, TableExtractionResult
from marie.extract.structures import SerializationManager, UnstructuredDocument
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import StructuredRegion
from marie.extract.validator.base import ValidationContext, ValidationSummary
from marie.logging_core.predefined import default_logger as logger
from marie.utils.docs import frames_from_file
from marie.utils.json import load_json_file
from marie.utils.overlap import merge_bboxes_as_block


@lru_cache(maxsize=None)
def _get_table_rec(model_path: str, **kwargs) -> TableRecPredictor:
    table_rec_predictor = TableRecPredictor(checkpoint=model_path)
    return table_rec_predictor


def check_directories_exist(directories: List[str]) -> None:
    """
    Validates the existence of required directories, raising an error if any is missing.
    """
    missing_dirs = [
        directory for directory in directories if not os.path.exists(directory)
    ]
    if missing_dirs:
        raise FileNotFoundError(
            f"The following directories are missing: {', '.join(missing_dirs)}"
        )


def parse_claims_to_document(
    doc: UnstructuredDocument,
    claims_dir: str,
) -> None:
    """
    Load pre-extracted claims and add them as annotations to the document.

    This follows the same pattern as process_extractions() but for pre-parsed claims.
    Adds CLAIM annotations to the document's lines, making the document the
    single source of truth for all annotations.

    Args:
        doc: The UnstructuredDocument to update
        claims_dir: Path to claims output directory (agent-output/claims)
    """
    if not os.path.exists(claims_dir):
        logger.debug(f"Claims directory does not exist: {claims_dir}")
        return

    files = sorted(f for f in os.listdir(claims_dir) if f.lower().endswith(".json"))

    if not files:
        logger.debug(f"No JSON files in claims directory: {claims_dir}")
        return

    logger.info(f"Parsing {len(files)} claim files to document")

    for file in files:
        try:
            page_id = extract_page_id(file) - 1  # Convert to 0-indexed
            if page_id < 0:
                logger.warning(f"Could not extract page id from file: {file}")
                continue

            json_path = os.path.join(claims_dir, file)
            json_result = load_json_file(json_path, safe_parse=True)

            if not json_result or "extractions" not in json_result:
                logger.debug(f"No extractions in claims file: {file}")
                continue

            # Create ExtractionResult and process like other parsers
            extraction = ExtractionResult(**json_result)

            # Get lines for this page
            lines_for_page = sorted(
                doc.lines_for_page(page_id), key=lambda ln: ln.metadata.line_id
            )

            # Add CLAIM annotations to lines
            for segment in extraction.extractions:
                row_number = int(segment.line_number)
                meta_line = locate_line(lines_for_page, row_number)

                if meta_line:
                    _annotate_segment("CLAIM", meta_line, segment)
                else:
                    logger.warning(f"No line found for page {page_id} row {row_number}")

            logger.debug(
                f"Added {len(extraction.extractions)} CLAIM annotations to page {page_id + 1}"
            )

        except Exception as e:
            logger.error(f"Error parsing claims file {file}: {e}")
            continue

    logger.info("Finished parsing claims to document")


def create_unstructured_doc(frames: list, metadata: dict) -> UnstructuredDocument:
    """Create and return an UnstructuredDocument object."""
    ocr_meta = metadata.get("ocr", [])
    if not ocr_meta:
        raise ValueError("OCR metadata is required but missing in source metadata.")
    visualize_icr(frames, ocr_meta)

    unstructured_meta = {
        'ref_id': metadata["ref_id"],
        'ref_type': metadata["ref_type"],
        'job_id': metadata["job_id"],
        'source_metadata': metadata,
    }
    return MetaReader.from_data(
        frames=frames, ocr_meta=ocr_meta, unstructured_meta=unstructured_meta
    )


def highlight_tables(doc: UnstructuredDocument, frames: list, output_dir: str):
    logger.info(f"Highlight tables")

    for page_id in range(doc.page_count):
        frame = frames[page_id]
        lines = doc.lines_for_page(page_id)
        # filter all the TABLE and TABLE create_unstructured_doc HEADER annotations
        collected_lines = []
        for line in lines:
            if line.annotations is None:
                continue
            for annotation in line.annotations:
                if annotation.annotation_type == "TABLE":
                    logger.debug(f"Annotation {annotation} > {line}")
                    collected_lines.append(line)

        # Highlight the lines in the image by getting the bounding box for highlighted lines
        viz_img = None
        if not isinstance(frame, Image.Image):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            viz_img = Image.fromarray(frame)
        else:
            viz_img = frame

        size = 14
        draw = ImageDraw.Draw(viz_img, "RGBA")  # Ensure RGBA mode for transparency
        font = get_font(size)
        for i, line in enumerate(collected_lines):
            logger.debug(f"Line {line} >> {line.metadata}")
            x, y, w, h = line.metadata.model.bbox
            box = line.metadata.model.bbox
            draw_box(
                draw,
                box,
                None,  # f"{q_text} : {q_confidence}",
                (255, 0, 0, 128),  # semi-transparent red
                font,
            )

        viz_img.save(f"{output_dir}/{page_id + 1}-ZZ.png")


def render_document_markdown(
    doc: UnstructuredDocument, output_file: str = "document.md"
):
    """
    Renders the `UnstructuredDocument` as a Markdown file and prints it to the console, showing annotated lines and their annotations.

    Parameters:
        doc (UnstructuredDocument): The document to render.
        output_file (str): The file path to write the Markdown output into. Defaults to "document.md".

    Returns:
        None
    """
    markdown_output = ["# Document in Markdown Format\n"]

    for page_id in range(doc.page_count):
        markdown_output.append(f"## Page {page_id + 1}\n")
        lines = doc.lines_for_page(page_id)
        regions = doc.regions_for_page(page_id)

        markdown_output.append(f"### Regions found : {len(regions)}\n")

        if not lines:
            markdown_output.append("_No content on this page._\n")
            continue

        for meta_line in lines:
            if meta_line.annotations is None or len(meta_line.annotations) == 0:
                continue
            text_content = meta_line.line if meta_line.line else "No text available"
            markdown_output.append(f"- **Line:** {text_content}")

            markdown_output.append("  - **Annotations:**")
            for annotation in meta_line.annotations:
                if isinstance(annotation, TypedAnnotation):
                    markdown_output.append(
                        # f"    -  Type: `{annotation.annotation_type}`, Name: `{annotation.name}`, Value: `{annotation.value}`, Start: {annotation.start}, End: {annotation.end}"
                        f"    -  Type: `{annotation.annotation_type}`, Name: `{annotation.name}`, Value: `{annotation.value}`"
                    )
                else:
                    raise ValueError(f"Unknown annotation type : {annotation}")
        markdown_output.append("\n")

    markdown_string = "\n".join(markdown_output)

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(markdown_string)
        logger.info(f"Markdown output written to {output_file}")
    except IOError as e:
        logger.error(f"Error writing to file {output_file}: {e}")


def correct_row_numbers(extraction_items: List[Segment]) -> List[Segment]:
    """
    Parses and validates the row_number and reason fields of ExtractionItem objects.
    If a mismatch is found between row_number and the row inferred from reason,
    the row inferred from reason is used as the ground truth to correct row_number.

    Parameters:
        extraction_items (List[ExtractionItem]): A list of ExtractionItem objects.

    Returns:
        List[ExtractionItem]: The corrected list of ExtractionItem objects.
    """
    row_number_pattern = re.compile(r"Found in row (\d+)")
    for item in extraction_items:
        match = row_number_pattern.search(item.label_found_at)
        if match:
            try:
                reason_row_number = int(match.group(1))
                if item.line_number != reason_row_number:
                    # we can correct the row_number but this requires more refinement
                    logging.warning(
                        f"Mismatch found: row_number={item.line_number}, reason_row={reason_row_number}. Correcting..."
                    )
                    # item.line_number = reason_row_number
            except ValueError:
                logging.warning(
                    f"Could not parse row number as integer: {match.group(1)}. Skipping correction."
                )
        else:
            logging.warning(
                f"Invalid reason format: {item.label_found_at}. Skipping this item."
            )
    return extraction_items


def parse_files(
    doc: UnstructuredDocument,
    working_dir: str,
    src_dir: str,
    update_fn: Callable[
        [UnstructuredDocument, Union[ExtractionResult, TableExtractionResult], int],
        None,
    ],
    log_prefix: str,
    result_type: Type[Union[ExtractionResult, TableExtractionResult]],
    conf: OmegaConf,
) -> None:
    """Helper function to parse files and update the document."""
    files = sorted(
        [file for file in os.listdir(src_dir) if file.lower().endswith(".json")]
    )
    logging.info(f"Starting parsing of {log_prefix.lower()} : {len(files)} json files")

    for file in files:
        try:
            page_id = (
                extract_page_id(file) - 1
            )  # Files indexed at 1, UnstructuredDocument indexed at 0
            logging.info(
                f"Processing {log_prefix} file: {file} with extracted page_id: {page_id}"
            )
            result_json_file = os.path.join(src_dir, file)

            json_result = load_json_file(result_json_file, safe_parse=True)
            if json_result is None:
                logging.warning(
                    f"File {result_json_file} does not contain any JSON data, skipping."
                )
                continue

            # Validate JSON content before creating the ExtractionResult or TableExtractionResult object
            if "extractions" not in json_result:
                logging.error(
                    f"File {result_json_file} is missing the required 'extractions' field. Skipping."
                )
                continue

            try:
                extraction = result_type(**json_result)
            except ValidationError as e:
                logging.error(
                    f"Validation error when creating {result_type.__name__} for file {result_json_file}: {e}"
                )
                raise e  # this is a critical error and we need to stop processing and fix the data

            update_fn(working_dir, doc, extraction, page_id, conf)
        except Exception as e:
            logging.error(f"Error processing {log_prefix} file {file}: {e}")
            continue


def _normalize_key(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    return re.sub(r"\s+", "_", label.strip().upper())


def process_extractions(
    working_dir: str,
    doc: UnstructuredDocument,
    result: ExtractionResult,
    page_id: int,
    annotation_type: str,
    grounding_keys: list[str],
    add_container: bool = True,
) -> None:
    """
    Processes and annotates extractions for a specific page and annotation type.

    Args:
        working_dir (str): The working directory.
        doc (UnstructuredDocument): The document being updated.
        result (ExtractionResult): The extraction result containing segments.
        page_id (int): The ID of the page to process.
        annotation_type (str): Type of annotation ("CLAIM", "REMARKS", etc.).
        grounding_keys (list[str]): List of valid keys to process.
        add_container (bool): Whether to add a container region for the annotations. Defaults to False.

    If add_container=True:
      - Emit {ANNOTATIONTYPE}_START before the first annotated segment.
      - Emit {ANNOTATIONTYPE}_END after the last annotated segment.
    """
    lines_for_page: list[LineWithMeta] = sorted(
        doc.lines_for_page(page_id), key=lambda ln: ln.metadata.line_id
    )
    logger.info(f"lines => {len(lines_for_page)} >")
    logger.info(f"Detailed extraction result for page {page_id}")

    first_emitted = False
    last_meta_line: LineWithMeta | None = None
    last_row_number: int | None = None

    extractions = correct_row_numbers(result.extractions)

    for segment in extractions:
        row_number = int(segment.line_number)
        segment.label = segment.label.upper() if segment.label else None
        key, value, reason = segment.label, segment.value, segment.reasoning

        # Normalize spaces into underscores
        key = re.sub(r"\s+", "_", key) if key else None
        if key not in grounding_keys:
            logging.warning(
                f"Not a grounded Key: {key}, Grounded Value: {value}, reason:\n{reason}"
            )
            continue

        if "ERROR" == key:
            logging.warning(f"Invalid key: {key} > {value}")
            continue

        meta_line = locate_line(lines_for_page, row_number)
        if meta_line is None:
            logging.warning(
                f"No line found for page {page_id} and row number {row_number}"
            )
            continue

        # Emit START once, right before annotating the first valid segment
        if add_container and not first_emitted:
            segment_start = Segment(
                line_number=row_number,
                label=f"{annotation_type}_START",
                value=f"{annotation_type.title()} Start Annotation",
                label_found_at=f"Start at {row_number}",
                reasoning=f"Automatically inserted {annotation_type.lower()} start",
            )
            _annotate_segment(annotation_type, meta_line, segment_start)
            first_emitted = True

        logger.info(f"   --> Line : {meta_line}")
        _annotate_segment(annotation_type, meta_line, segment)

        last_meta_line = meta_line
        last_row_number = row_number

    if (
        add_container
        and first_emitted
        and last_meta_line is not None
        and last_row_number is not None
    ):
        segment_end = Segment(
            line_number=last_row_number,
            label=f"{annotation_type}_END",
            value=f"{annotation_type.title()} End Annotation",
            label_found_at=f"End at {last_row_number}",
            reasoning=f"Automatically inserted {annotation_type.lower()} end",
        )
        _annotate_segment(annotation_type, last_meta_line, segment_end)


def extract_tables(
    doc: UnstructuredDocument, frames: list, metadata: dict[str, any], output_dir: str
):
    YELLOW_COLOR = (0, 255, 255)  # OpenCV uses BGR, so this represents yellow
    logger.info("Extracting tables...")
    model_path = os.path.join(__model_path__, "table_recognition", "2025_02_18")
    # table_rec_predictor = TableRecPredictor(checkpoint=model_path)
    table_rec_predictor = _get_table_rec(model_path)

    for page_id in range(doc.page_count):
        logger.info(f"Page {page_id + 1}")
        frame = frames[page_id]
        lines = doc.lines_for_page(page_id)
        all_tables = []
        table_lines = []
        collecting = False

        for line in lines:
            if line.annotations is None:
                continue
            for annotation in line.annotations:
                if (
                    annotation.annotation_type == "TABLE"
                    and annotation.name == "TABLE_START"
                ):
                    if collecting:
                        logger.warning(
                            f"Warning: Nested TABLE_START found without TABLE_END. Closing previous table."
                        )
                        all_tables.append(table_lines)
                        table_lines = []
                    collecting = True
                    logger.info(f"Table start found: {annotation} > {line}")
                if collecting and annotation.name not in ["TABLE_START", "TABLE_END"]:
                    table_lines.append(line)
                if (
                    annotation.annotation_type == "TABLE"
                    and annotation.name == "TABLE_END"
                ):
                    if collecting:
                        logger.info(f"Table end found: {annotation} > {line}")
                        all_tables.append(table_lines)
                        table_lines = []
                        collecting = False
                    else:
                        logger.warning(
                            f"Warning: TABLE_END found without matching TABLE_START. Skipping."
                        )
        if collecting:
            print(
                f"Warning: TABLE_START found without TABLE_END at the end of the page. Closing table."
            )
            all_tables.append(table_lines)
        print(f"Collected {len(all_tables)} Table(s) for Page {page_id + 1}")

        # Output each table in a nice formatted structure
        for i, table_lines in enumerate(all_tables):
            print(f"\nTable {i + 1} (Page {page_id + 1}):")
            for line in table_lines:
                text = line.line.strip() if line.line else ""
                annotations = ", ".join(
                    [
                        f"{ann.annotation_type}:{ann.name}"
                        for ann in (line.annotations or [])
                    ]
                )
                print(f"  {text:<50} | {annotations}")

            table_bboxes = [t.metadata.model.bbox for t in table_lines]

            overlay = Image.fromarray(frame)
            draw = ImageDraw.Draw(overlay, "RGBA")
            for t in table_lines:
                for word in t.metadata.model.words:
                    x, y, w, h = word.box
                    draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 0, 70))
            # overlay.save(os.path.join(output_dir, "fragments", f"blended_{page_id + 1}_{i}.png"))

            table_bbox = merge_bboxes_as_block(table_bboxes)
            x, y, w, h = table_bbox

            snippet = np.array(overlay)[y : y + h, x : x + w :]
            snippet = cv2.cvtColor(snippet, cv2.COLOR_RGBA2RGB)
            snippet = frame[y : y + h, x : x + w :]
            pil_snippet = Image.fromarray(snippet)
            results = table_rec_predictor([pil_snippet])
            print("Table recognition results:")

            if len(results) == 1:
                cols = results[0].cols
                draw = ImageDraw.Draw(pil_snippet, "RGBA")
                distinct_colors = generate_distinct_colors()
                color_cycle = cycle(distinct_colors)
                for item in cols:
                    bbox = (item.bbox[0], 0, item.bbox[2], pil_snippet.size[1])
                    color = (*next(color_cycle), 128)
                    draw.rectangle(bbox, fill=color, outline=color)
                pil_snippet.save(
                    os.path.join(output_dir, "fragments", f"{page_id + 1}_{i}.png")
                )

            if False:
                cv2.imwrite(
                    os.path.join(output_dir, "fragments", f"{page_id + 1}_{i}.png"),
                    snippet,
                )

            # write table_lines to a file
            with open(
                os.path.join(
                    output_dir, "fragments", f"{page_id + 1}_{i}_INJECTED_TEXT.txt"
                ),
                "w",
            ) as f:
                # f.write(f"Page dimensions: {table_bbox[2]}x{table_bbox[3]}\n")
                # f.write(f"[Image 0x0 to {table_bbox[2]}x{table_bbox[3]}]\n")
                if False:
                    for line in table_lines:
                        # Normalize the line's coordinates relative to table_bbox
                        words = line.metadata.model.words  # WordModel
                        for word in words:
                            box = word.box
                            x = box[0] - table_bbox[0]
                            y = box[1] - table_bbox[1]
                            w = box[2]
                            h = box[3]
                            f.write(f"[{x},{y},{w},{h}]{word.text}\n")

                # TODO : Quantize using lmdx
                if True:
                    try:
                        from marie.components.document_taxonomy.verbalizers import (
                            verbalizers,
                        )

                        # adding spatial context
                        metadata_ocr = doc.source_metadata["ocr"]
                        verb_lines = []
                        for k, meta in enumerate(metadata_ocr):
                            page_number = meta["meta"]["page"]
                            if page_id != page_number:
                                continue
                            verb_lines = verbalizers("SPATIAL_FORMAT", meta)
                            break

                        for line in table_lines:
                            line_number = line.metadata.line_id
                            text_line = (
                                verb_lines[line_number - 1]
                                if line_number <= len(verb_lines)
                                else None
                            )
                            f.write(f"{text_line['text']}\n")
                    except Exception as e:
                        print(f"Error while writing lines: {e}")
                        continue


def process_extractions_table(
    working_dir: str,
    doc: UnstructuredDocument,
    result: TableExtractionResult,
    page_id: int,
    annotation_type: str,
    grounding_keys: list[str],
    conf: OmegaConf,
    **kwargs,
) -> None:
    """
    Processes and annotates extractions for a specific page and annotation type.

    Args:
        working_dir (str): The working directory where the files are located.
        doc (UnstructuredDocument): The document being updated.
        result (TableExtractionResult): The extraction result containing table-related data.
        page_id (int): The ID of the page to process.
        annotation_type (str): Type of annotation ("CLAIM", "REMARKS", etc.).
        grounding_keys (list[str]): List of valid keys to process.
        conf (OmegaConf): Configuration object for additional settings.
    """
    logger.info(f"Detailed extraction result for page {page_id}")

    lines_for_page: list[LineWithMeta] = sorted(
        doc.lines_for_page(page_id), key=lambda ln: ln.metadata.line_id
    )
    extractions = result.extractions
    table_segment_meta = {}
    table_start_lines = []
    fill_missing_row_gaps = True

    # Step 1 - Create annotations

    for segment_index, segment in enumerate(extractions):
        logger.info(f'Table segment page {page_id}, table seg # {segment_index}')
        header_rows = segment.header_rows
        rows = segment.rows

        mapping = OrderedDict()
        mapping["TABLE_HEADER"] = header_rows
        mapping["TABLE_ROWS"] = rows

        # Root label is TABLE_START and then TABLE_END
        # Insert TABLE_START annotation at the very beginning of the table
        first_row_number = 0
        table_start_line = None
        table_end_line = None

        if header_rows:  # If there is at least one header_row
            first_row_number = int(header_rows[0].line_number)
            table_start_line = locate_line(lines_for_page, first_row_number)
            table_start_lines.append(table_start_line)

            if table_start_line:
                segment_start = Segment(
                    line_number=first_row_number,
                    label="TABLE_START",
                    value="Table Start Annotation",
                    label_found_at=f"Start at {first_row_number}",
                    reasoning="Automatically inserted table start",
                )
                _annotate_segment(annotation_type, table_start_line, segment_start)

        for key, rows in mapping.items():
            missing_line_segments = []
            if fill_missing_row_gaps and key == "TABLE_ROWS" and rows:
                logging.warning(
                    f"Inserting gap segment on page_id {page_id}, table # {segment_index} ..."
                )
                all_row_lines = sorted(int(row.line_number) for row in rows)
                full_range = range(all_row_lines[0], all_row_lines[-1] + 1)
                existing_lines = {int(row.line_number) for row in rows}

                for ln in full_range:
                    if ln not in existing_lines:
                        logging.warning(
                            f"Missing line {ln} in {key} for page {page_id}, inserting gap segment."
                        )
                        meta_line = locate_line(lines_for_page, ln)
                        if meta_line:
                            gap_segment = Segment(
                                line_number=ln,
                                label=key,
                                value="[GAP]",
                                label_found_at=f"Inserted missing row line {ln}",
                                reasoning="Auto-filled missing line between detected table rows",
                            )
                            missing_line_segments.append((ln, meta_line, gap_segment))

            for row in rows:
                row_number = int(row.line_number)
                key, value, reason = key, row.value, row.reasoning

                if "ERROR" == key:
                    logging.warning(f"Invalid key: {key} > {value}")
                    continue

                meta_line = locate_line(lines_for_page, row_number)
                if meta_line is None:
                    logging.warning(
                        f"No line found for page {page_id} and row number {row_number}"
                    )
                    continue

                print(f"   --> Line : {meta_line}")
                segment = Segment(
                    line_number=row_number,
                    label=key,
                    value=value,
                    label_found_at=f"Found at {row_number}",
                    reasoning=reason,
                )
                _annotate_segment(annotation_type, meta_line, segment)

            # Annotate any missing gaps after processing real rows
            for ln, meta_line, gap_segment in missing_line_segments:
                print(f"   --> Line (gap) : {meta_line}")
                _annotate_segment(annotation_type, meta_line, gap_segment)

        if header_rows:  # If there is at least one header_row
            # Insert TABLE_END annotation at the very end of the table
            last_row_number = int(rows[-1].line_number) if rows else first_row_number
            table_end_line = locate_line(lines_for_page, last_row_number)
            if table_end_line:
                segment_end = Segment(
                    line_number=last_row_number,
                    label="TABLE_END",
                    value="Table End Annotation",
                    label_found_at=f"End at {last_row_number}",
                    reasoning="Automatically inserted table end",
                )
                _annotate_segment(annotation_type, table_end_line, segment_end)

        table_segment_meta[segment_index] = {
            "start": table_start_line,
            "end": table_end_line,
        }

    logger.info('' + '-' * 50)
    logger.info(f"Total number of table segments: {len(extractions)}")

    # Step 2 - process extractions

    extracted_tables_src_dir = os.path.join(
        working_dir, "agent-output", "table-extract"
    )

    for segment_index, segment in enumerate(extractions):
        logger.info(f'Table segment page {page_id}, table seg # {segment_index}')
        file_page_id = page_id + 1  # files are named with 1 - based indexes

        # Check if Markdown file of the format PAGENUM_SEGMENT.png.md exists
        file_name = f"{file_page_id}_{segment_index}.png.md"
        file_name = os.path.join(extracted_tables_src_dir, file_name)

        if os.path.exists(file_name):
            logger.info(f"Markdown file `{file_name}` exists, loading it.")
        else:
            logger.info(f"Markdown file `{file_name}` does not exist.")
            continue

        md_content = None
        with open(file_name, "r") as md_file:
            md_content = md_file.read()

        # TODO : Today we only support layer-main
        #        Add support for multiple layers
        layer_config = conf.layers['layer-main']

        try:
            parsing_method = kwargs.get("parsing_method", "mrp")
            logger.info(f"Table parsing method: {parsing_method}")
            region: Optional[StructuredRegion] = None

            if parsing_method == "plain":
                region = _parse_table_plain(
                    doc,
                    md_content,
                    page_id,
                    segment_index,
                    table_segment_meta[segment_index],
                )
            elif parsing_method == "mrp":
                mp_cfg = OmegaConf.to_container(
                    layer_config.get("region_parser") or {}, resolve=True
                )

                region = _parse_table_mrp(
                    doc,
                    md_content,
                    page_id,
                    segment_index,
                    table_segment_meta[segment_index],
                    cfg=mp_cfg,
                )
            if region:
                #  source tracking
                region.tags.update(
                    {"source": 'result-parser', "processor_generated": False}
                )

                doc.insert_region(region)
        except Exception as e:
            logger.error(
                f"Failed to extract table from file: {file_name}", exc_info=True
            )
            raise e
        else:
            print(f"Markdown file `{file_name}` does not exist.")


def render_document_markdown_structured(
    doc: UnstructuredDocument, output_file: str = "structured.md"
):
    """
    Renders the `UnstructuredDocument` as a Markdown file and prints it to the console,
    showing annotated lines and their annotations with aligned columns for Type, Name, and Value.

    Parameters:
        doc (UnstructuredDocument): The document to render.
        output_file (str): The file path to write the Markdown output into. Defaults to "structured.md".

    Returns:
        None
    """
    markdown_output = []

    # Collect all annotations for alignment
    all_annotations = []
    for page_id in range(doc.page_count):
        lines = doc.lines_for_page(page_id)
        if not lines:
            continue
        page_annotations = []
        for meta_line in lines:
            if meta_line.annotations is None or len(meta_line.annotations) == 0:
                continue

            for annotation in meta_line.annotations:
                if isinstance(annotation, TypedAnnotation):
                    page_annotations.append(
                        {
                            "Type": annotation.annotation_type,
                            "Name": f"{annotation.name}",
                            "Value": f"{meta_line.metadata.line_id} | {annotation.value}",
                        }
                    )
                else:
                    raise ValueError(f"Unknown annotation type : {annotation}")

        if len(page_annotations) > 0:
            page_annotations.insert(
                0,
                {
                    "Type": "PAGE",
                    "Name": "PAGE_ID",
                    "Value": f"{page_id}",
                },
            )
            all_annotations.extend(page_annotations)

    if not all_annotations:
        logging.warning("No annotations found in the document.")
        return

    # Calculate column widths to align content
    type_width = max(len("Type"), max(len(a["Type"]) for a in all_annotations)) + 2
    name_width = max(len("Name"), max(len(a["Name"]) for a in all_annotations)) + 2
    value_width = max(len("Value"), max(len(a["Value"]) for a in all_annotations)) + 2

    # Format and append annotations to markdown output
    for annotation in all_annotations:
        markdown_output.append(
            f"-  Type: `{annotation['Type']:<{type_width}}`"
            f"Name: `{annotation['Name']:<{name_width}}`"
            f"Value: `{annotation['Value']}`"
        )
    markdown_output.append("\n")
    markdown_string = "\n".join(markdown_output)

    logging.debug(markdown_string)
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(markdown_string)
        logging.info(f"Markdown output written to {output_file}")
    except IOError as e:
        logging.error(f"Error writing to file {output_file}: {e}")


def parse_results(working_dir: str, metadata: dict, conf: OmegaConf) -> None:
    """
    Main entry point for parsing results. Executes all registered parsers and optionally validates their outputs.

    :param working_dir: The directory containing the working files.
    :param metadata: Metadata associated with the document.
    :param conf: Configuration object specifying annotators and processing options.
    :raises ValueError: If the specified working directory does not exist.
    """

    logging.info(f"Parsing results from {working_dir}")
    if not os.path.exists(working_dir):
        raise ValueError(f"'{working_dir}' does not exist")

    agent_output_dir = os.path.join(working_dir, "agent-output")
    output_dir = os.path.join(working_dir, "parsed-result")
    frames_dir = os.path.join(working_dir, "frames")

    os.makedirs(agent_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    DIRS = {
        name: os.path.join(agent_output_dir, name) for name in conf.annotators.keys()
    }
    check_directories_exist(DIRS.values())

    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".png"))
    frames = [frames_from_file(os.path.join(frames_dir, f))[0] for f in files]
    doc: UnstructuredDocument = create_unstructured_doc(frames, metadata)

    all_validation_summaries = []
    validation_enabled = conf.get("validation", {}).get("enabled", True)
    fail_on_validation_errors = conf.get("validation", {}).get("fail_on_errors", False)

    logger.info(f"Validation  enabled: {validation_enabled}")
    logger.info(f"Fail on validation errors: {fail_on_validation_errors}")

    for name, ann_conf in conf.annotators.items():
        target = ann_conf.get("parser", name)
        parser_fn = component_registry.get_parser(target)

        if not parser_fn:
            logging.error(f"No parser registered for '{target}'")
            raise ValueError(f"Parser '{target}' not found in registry")

        try:
            logging.info(f"Running parser '{target}' for '{name}'")
            parser_fn(doc, working_dir, DIRS[name], conf)

            # Validate parser output if validation is enabled
            if validation_enabled:
                logging.info(f"Validating output from parser '{target}'")

                # Get validator names from config or use defaults
                validator_names = ann_conf.get("validators", None)
                if validator_names is None:
                    # Use validation configuration or default to all validators that support the stage
                    validator_names = conf.get("validation", {}).get(
                        "default_validators", None
                    )

                # Run validation
                validation_summary: ValidationSummary = validate_parser_output(
                    doc=doc,
                    parser_name=target,
                    working_dir=working_dir,
                    src_dir=DIRS[name],
                    conf=conf,
                    validator_names=validator_names,
                    metadata=metadata,
                    frames=frames,
                )

                all_validation_summaries.append(validation_summary)

                # Log validation results
                if validation_summary.overall_valid:
                    logging.info(
                        f"✓ Parser '{target}' validation passed ({validation_summary.total_warnings} warnings)"
                    )
                else:
                    logging.error(
                        f"✗ Parser '{target}' validation failed ({validation_summary.total_errors} errors, {validation_summary.total_warnings} warnings)"
                    )

                    # Log specific errors and warnings
                    for result in validation_summary.results:
                        for error in result.errors:
                            logging.error(f"  {result.validator_name}: {error}")
                        for warning in result.warnings:
                            logging.warning(f"  {result.validator_name}: {warning}")

                    # Optionally fail fast on validation errors
                    if fail_on_validation_errors:
                        critical_errors = validation_summary.get_errors_by_severity(
                            "CRITICAL"
                        )
                        if critical_errors:
                            raise ValueError(
                                f"Critical validation errors in parser '{target}': {[str(e) for e in critical_errors]}"
                            )

                        # Or fail on any errors if configured
                        if validation_summary.total_errors > 0:
                            raise ValueError(
                                f"Validation failed for parser '{target}' with {validation_summary.total_errors} errors"
                            )

        except Exception as e:
            logging.error(f"Error in parser '{target}': {e}")
            raise e

    if conf.get("processing", {}).get("render_markdown", False):
        render_document_markdown(doc, os.path.join(output_dir, "document.md"))
        render_document_markdown_structured(
            doc, os.path.join(output_dir, "structured.md")
        )
        SerializationManager.serialize(doc, os.path.join(output_dir, "document.pkl"))

    logging.info("Converting document to structured output")
    results: SubzeroResult = convert_document_to_structure(doc, conf, output_dir)

    # Validate converter output
    if validation_enabled:
        logging.info("Validating converter output")

        # Get converter validators from config
        converter_validators = conf.get("validation", {}).get(
            "converter_validators", None
        )

        converter_context = ValidationContext.create_converter_context(
            original_doc=doc,
            converter_output=results,
            working_dir=working_dir,
            conf=conf,
            metadata=metadata,
        )

        converter_validation = validate_document(
            # doc=results,  # The converter output
            context=converter_context,
            validator_names=converter_validators,
        )

        all_validation_summaries.append(converter_validation)

        # Log converter validation results
        if converter_validation.overall_valid:
            logging.info(
                f"✓ Converter validation passed ({converter_validation.total_warnings} warnings)"
            )
        else:
            logging.error(
                f"✗ Converter validation failed ({converter_validation.total_errors} errors, {converter_validation.total_warnings} warnings)"
            )

            for result in converter_validation.results:
                for error in result.errors:
                    logging.error(f"  {result.validator_name}: {error}")
                for warning in result.warnings:
                    logging.warning(f"  {result.validator_name}: {warning}")

            if fail_on_validation_errors:
                critical_errors = converter_validation.get_errors_by_severity(
                    "CRITICAL"
                )
                if critical_errors:
                    raise ValueError(
                        f"Critical validation errors in converter output: {[str(e) for e in critical_errors]}"
                    )

                if converter_validation.total_errors > 0:
                    raise ValueError(
                        f"Converter validation failed with {converter_validation.total_errors} errors"
                    )

    if validation_enabled and all_validation_summaries:
        generate_validation_report(all_validation_summaries, output_dir)
