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
                f"Not a grounded Key:\n"
                f"  Key: {key}\n"
                f"  Annotation Type: {annotation_type}\n"
                f"  Valid Keys: {grounding_keys}\n"
                f"  Grounded Value: {value}\n"
                f"  Reason: {reason}"
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
    raise NotImplementedError("extract_tables is deprecated, use table parser instead")
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


def _run_region_builder_if_configured(
    doc: UnstructuredDocument, working_dir: str, conf: OmegaConf
) -> None:
    """
    Run the region-builder parser if any layer has a data_source configured.

    Region-builder creates StructuredRegion objects from JSON data sources
    (like claim-extract) and inserts them into the document.
    """
    layers_conf = conf.get("layers", {})
    if not layers_conf:
        return

    # Check if any layer has region_parser.data_source configured
    has_data_source = False
    for layer_name, layer_conf in layers_conf.items():
        region_parser_conf = layer_conf.get("region_parser", {})
        if region_parser_conf.get("data_source"):
            has_data_source = True
            break

    if not has_data_source:
        logging.debug("No data_source configured in any layer, skipping region-builder")
        return

    # Get the region-builder parser
    region_builder_fn = component_registry.get_parser("region-builder")
    if not region_builder_fn:
        logging.warning(
            "region-builder parser not registered but data_source is configured. "
        )
        return

    try:
        logging.info("Running region-builder to create StructuredRegion objects")
        # src_dir is not used by region-builder (it reads from data_source in config)
        region_builder_fn(doc, working_dir, "", conf)
    except Exception as e:
        logging.error(f"Error in region-builder: {e}")
        raise


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

    print(f'conf : {conf}')

    DIRS = {
        name: os.path.join(agent_output_dir, name) for name in conf.annotators.keys()
    }
    print(f"Expected directories for annotators: {DIRS}")
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
                        f"Parser '{target}' validation passed ({validation_summary.total_warnings} warnings)"
                    )
                else:
                    logging.error(
                        f"Parser '{target}' validation failed ({validation_summary.total_errors} errors, {validation_summary.total_warnings} warnings)"
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

    # Run region-builder if any layer has data_source configured
    # Region-builder creates StructuredRegion objects from JSON data sources
    _run_region_builder_if_configured(doc, working_dir, conf)

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

        if converter_validation.overall_valid:
            logging.info(
                f"Converter validation passed ({converter_validation.total_warnings} warnings)"
            )
        else:
            logging.error(
                f"Converter validation failed ({converter_validation.total_errors} errors, {converter_validation.total_warnings} warnings)"
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
