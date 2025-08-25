import logging
import os

from omegaconf import OmegaConf

from marie.extract.results.base import _annotate_segment, locate_line
from marie.extract.results.registry import register_parser
from marie.extract.results.result_parser import (
    parse_files,
    process_extractions,
    process_extractions_table,
)
from marie.extract.schema import ExtractionResult, Segment, TableExtractionResult
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.utils.json import load_json_file


@register_parser("noop")
def parse_noop(
    doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
) -> None:
    """
    No-op parser for annotations. This is a placeholder and does not perform any action.
    """
    logging.info("No-op parser for annotations called. No action taken.")
    # This function is intentionally left empty to serve as a placeholder.
    pass


@register_parser("key-values")
def parse_key_value(
    doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
) -> None:
    """Parse KV from JSON files and update the document."""
    parse_files(
        doc,
        working_dir,
        src_dir,
        update_document_kv,
        "Key-Values",
        result_type=ExtractionResult,
        conf=conf,
    )


@register_parser("annotations")
def parse_annotations(
    doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
) -> None:
    """
    Parse a single annotations.json file in `src_dir` and annotate each line.

    Expected format:
      [
        {
          "page_id": 0,
          "page_number": 1,
          "extractions": [
            {
              "line_number": 1,
              "label": "...",
              "value": "...",
              "label_found_at": "...",
              "reasoning": "...",
              "source_text": "..."
            },
            ...
          ]
        },
        ...
      ]
    """
    path = os.path.join(src_dir, "annotations.json")
    if not os.path.isfile(path):
        logging.info("No annotations.json in %s → skipping", src_dir)
        return

    try:
        pages = load_json_file(path)
    except Exception as e:
        logging.error("Failed to load %s: %s", path, e)
        return

    for page in pages:
        page_id = page.get("page_id")
        if page_id is None:
            logging.warning("Skipping entry without page_id: %s", page)
            continue

        lines: list[LineWithMeta] = doc.lines_for_page(page_id)
        for ext in page.get("extractions", []):
            ln = ext.get("line_number")
            if ln is None:
                logging.warning("Missing line_number in extraction: %s", ext)
                continue

            meta_line = locate_line(lines, ln)
            if meta_line is None:
                logging.warning(
                    "No line %d on page %d for annotation %r", ln, page_id, ext
                )
                continue

            segment = Segment(
                line_number=ln,
                label=ext["label"],
                value=ext["value"],
                label_found_at=ext["label_found_at"],
                reasoning=ext.get("reasoning", ""),
            )
            # annotate with a generic type; adjust if you want to use a specific annotation_type
            _annotate_segment("ANNOTATION", meta_line, segment)

    logging.info("Finished parsing annotations.json in %s", src_dir)


@register_parser("tables")
def parse_tables(
    doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
) -> None:
    """Parse Table from JSON files and update the document."""
    parse_files(
        doc,
        working_dir,
        src_dir,
        update_document_tables,
        "Tables",
        result_type=TableExtractionResult,
        conf=conf,
    )


def update_document_kv(
    working_dir: str,
    doc: UnstructuredDocument,
    result: ExtractionResult,
    page_id: int,
    conf: OmegaConf,
) -> None:
    """
    Updates the document with extracted key-value related data for the specified page.
    """
    process_extractions(
        working_dir, doc, result, page_id, "KV", conf.grounding["key-value"]
    )


def update_document_tables(
    working_dir,
    doc: UnstructuredDocument,
    result: ExtractionResult,
    page_id: int,
    conf: OmegaConf,
) -> None:
    """
    Updates the document with extracted remark-related data for the specified page.
    """
    process_extractions_table(
        working_dir, doc, result, page_id, "TABLE", conf.grounding["table"], conf
    )
