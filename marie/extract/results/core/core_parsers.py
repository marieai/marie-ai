import logging
import os
from typing import Dict, List

from omegaconf import OmegaConf

from marie.extract.parser.json_region_parser import JsonRegionParser
from marie.extract.registry import register_parser
from marie.extract.results.base import _annotate_segment, extract_page_id, locate_line
from marie.extract.results.result_parser import (
    parse_files,
    process_extractions,
)
from marie.extract.schema import ExtractionResult, Segment, TableExtractionResult
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import StructuredRegion
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
    """Parse Table from JSON files and update the document.

    Adapts table JSON format to TableExtractionResult by adding missing 'value' field
    using actual line content from the document when not present.
    """
    if not os.path.isdir(src_dir):
        logging.warning(f"tables source directory not found: {src_dir}")
        return

    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith('.json')])
    logging.info(f"Starting parsing of tables: {len(files)} json files")

    for file in files:
        try:
            page_id = extract_page_id(file) - 1  # Files indexed at 1, doc indexed at 0
            logging.info(f"Processing Tables file: {file} with page_id: {page_id}")

            result_json_file = os.path.join(src_dir, file)
            json_result = load_json_file(result_json_file, safe_parse=True)

            if json_result is None:
                logging.warning(f"File {result_json_file} has no JSON data, skipping.")
                continue

            if "extractions" not in json_result:
                logging.error(
                    f"File {result_json_file} missing 'extractions' field. Skipping."
                )
                continue

            # Adapt JSON to add missing 'value' field from document lines
            adapted_json = _adapt_tables_to_table_extraction_result(
                json_result, doc, page_id
            )

            extraction = TableExtractionResult(**adapted_json)
            update_document_tables(working_dir, doc, extraction, page_id, conf)

        except Exception as e:
            logging.error(f"Error processing Tables file {file}: {e}")
            raise


def _adapt_tables_to_table_extraction_result(
    data: dict, doc: UnstructuredDocument, page_id: int
) -> dict:
    """Transform tables JSON to TableExtractionResult format.

    Adds missing 'value' field by looking up line content from the document.
    If 'value' is already present, it is preserved.
    """
    # Build line lookup for this page using line_id
    lines_for_page = {ln.metadata.line_id: ln for ln in doc.lines_for_page(page_id)}

    def get_line_value(line_number: int) -> str:
        """Get the text content for a line number."""
        line = lines_for_page.get(line_number)
        if line:
            return line.line  # LineWithMeta uses .line, not .text
        return ""

    extractions = data.get('extractions', [])
    adapted_extractions = []

    for table in extractions:
        adapted_header_rows = []
        for row in table.get('header_rows', []):
            line_number = row.get('line_number', 0)
            # Preserve existing value if present, otherwise lookup from document
            value = row.get('value') if 'value' in row else get_line_value(line_number)
            adapted_header_rows.append(
                {
                    'line_number': line_number,
                    'value': value,
                    'found_at': row.get('found_at', f"Found in row {line_number}"),
                    'reasoning': row.get('reasoning'),
                }
            )

        adapted_rows = []
        for row in table.get('rows', []):
            line_number = row.get('line_number', 0)
            # Preserve existing value if present, otherwise lookup from document
            value = row.get('value') if 'value' in row else get_line_value(line_number)
            adapted_rows.append(
                {
                    'line_number': line_number,
                    'value': value,
                    'found_at': row.get('found_at', f"Found in row {line_number}"),
                    'reasoning': row.get('reasoning'),
                }
            )

        # Build continuation dict if present
        continuation = None
        if 'continuation' in table:
            cont = table['continuation']
            continuation = {
                'is_continuation': cont.get('is_continuation', False),
                'from_table_name': cont.get('from_table_name', ''),
                'continuation_rationale': cont.get('continuation_rationale', ''),
            }

        adapted_extractions.append(
            {
                'name': table.get('name', ''),
                'header_rows': adapted_header_rows,
                'rows': adapted_rows,
                'columns': table.get('columns', []),
                # Extended fields for classification and filtering
                'table_classification': table.get('table_classification'),
                'page_index': table.get('page_index'),
                'header_present': table.get('header_present'),
                'continuation': continuation,
                'columns_inferred': table.get('columns_inferred'),
            }
        )

    return {'extractions': adapted_extractions}


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
    result: TableExtractionResult,
    page_id: int,
    conf: OmegaConf,
) -> None:
    """
    Updates the document with table boundary annotations.

    Creates TABLE_START and TABLE_END annotations to mark table locations.
    Includes table_classification for filtering (e.g., 'CLAIM', 'REMARK', 'SUMMARY').
    Actual table data extraction is handled by 'claim-extract' parser.
    """
    lines_for_page = sorted(
        doc.lines_for_page(page_id), key=lambda ln: ln.metadata.line_id
    )

    extractions = result.extractions
    logging.info(f"Processing {len(extractions)} table(s) for page {page_id}")

    for segment_index, table in enumerate(extractions):
        header_rows = table.header_rows
        rows = table.rows

        # Get table classification for filtering
        classification = table.table_classification or "UNKNOWN"

        # Determine table boundaries
        first_row_number = None
        last_row_number = None

        if header_rows:
            first_row_number = int(header_rows[0].line_number)
        elif rows:
            first_row_number = int(rows[0].line_number)

        if rows:
            last_row_number = int(rows[-1].line_number)
        elif header_rows:
            last_row_number = int(header_rows[-1].line_number)

        if first_row_number is None:
            logging.warning(f"Table {segment_index} has no rows, skipping")
            continue

        # Use classification-specific annotation type (e.g., TABLE_CLAIM, TABLE_REMARK)
        annotation_type = f"TABLE_{classification}"

        # Create TABLE_START annotation
        table_start_line = locate_line(lines_for_page, first_row_number)
        if table_start_line:
            segment_start = Segment(
                line_number=first_row_number,
                label="TABLE_START",
                value=table.name or f"Table {segment_index}",
                label_found_at=f"Found in row {first_row_number}",
                reasoning=f"Table start at line {first_row_number}, classification: {classification}",
            )
            _annotate_segment(annotation_type, table_start_line, segment_start)

        # Create TABLE_END annotation
        if last_row_number:
            table_end_line = locate_line(lines_for_page, last_row_number)
            if table_end_line:
                segment_end = Segment(
                    line_number=last_row_number,
                    label="TABLE_END",
                    value=table.name or f"Table {segment_index}",
                    label_found_at=f"Found in row {last_row_number}",
                    reasoning=f"Table end at line {last_row_number}, classification: {classification}",
                )
                _annotate_segment(annotation_type, table_end_line, segment_end)

        logging.info(
            f"Table {segment_index} [{classification}]: lines {first_row_number}-{last_row_number}, "
            f"columns: {table.columns}"
        )


@register_parser("region-builder")
def parse_region_builder(
    doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
) -> None:
    """
    Build StructuredRegion objects from JSON data using config-driven section mapping.

    Reads the `data_source` config from each layer's region_parser configuration
    and creates regions from the corresponding JSON files in agent-output/{data_source}/.

    The section titles and parse types are read from region_parser.sections config:
        region_parser:
          data_source: claim-extract  # folder name in agent-output/
          sections:
            - title: CLAIM INFORMATION
              role: claim_information
              parse: kv
            - title: SERVICE LINES
              role: service_lines
              parse: table
            - title: CLAIM TOTALS
              role: claim_totals
              parse: kv
    """
    agent_output_dir = os.path.join(working_dir, "agent-output")

    # Get layers configuration
    layers_conf = conf.get("layers", {})
    if not layers_conf:
        logging.info("No layers configured, skipping region-builder")
        return

    total_regions = 0

    for layer_name, layer_conf in layers_conf.items():
        region_parser_conf = layer_conf.get("region_parser", {})
        data_source = region_parser_conf.get("data_source")

        if not data_source:
            logging.debug(
                f"Layer '{layer_name}' has no data_source configured, skipping"
            )
            continue

        data_source_dir = os.path.join(agent_output_dir, data_source)
        if not os.path.isdir(data_source_dir):
            logging.warning(f"Data source directory not found: {data_source_dir}")
            continue

        # Build section config from region_parser.sections
        sections_conf = region_parser_conf.get("sections", [])
        section_config = _build_region_section_config(sections_conf)

        # Process JSON files from the data source directory
        json_files = sorted(
            f for f in os.listdir(data_source_dir) if f.endswith('.json')
        )
        logging.info(f"Building regions from {len(json_files)} files in {data_source}")

        for json_file in json_files:
            file_path = os.path.join(data_source_dir, json_file)
            try:
                regions = _build_regions_from_json(
                    doc, file_path, layer_name, section_config
                )
                for region in regions:
                    doc.insert_region(region)
                    total_regions += 1
            except Exception as e:
                logging.error(f"Error building regions from {json_file}: {e}")
                continue

    logging.info(f"Region-builder created {total_regions} regions")


def _build_region_section_config(sections_conf: list) -> Dict:
    """
    Build section configuration from region_parser.sections config.

    Returns dict with:
        - kv_section_titles: list of section titles to parse as KV
        - table_section_titles: list of section titles to parse as table
        - role_mapping: dict mapping title -> role
        - json_key_mapping: dict mapping JSON keys to section titles
    """
    kv_titles = []
    table_titles = []
    role_mapping = {}
    # Map common JSON keys to section titles (case-insensitive)
    json_key_mapping = {}

    for section in sections_conf:
        title = section.get("title", "")
        role = section.get("role", "")
        parse_type = section.get("parse", "kv")

        title_upper = title.upper()
        role_mapping[title_upper] = role

        if parse_type == "table":
            table_titles.append(title_upper)
        else:
            kv_titles.append(title_upper)

        # Map role to title for JSON key lookup
        # e.g., service_lines -> SERVICE LINES, claim_information -> CLAIM INFORMATION
        if role:
            json_key_mapping[role.lower()] = title_upper
            # Also handle underscore variants
            json_key_mapping[role.lower().replace("_", " ")] = title_upper

    return {
        "kv_section_titles": kv_titles,
        "table_section_titles": table_titles,
        "role_mapping": role_mapping,
        "json_key_mapping": json_key_mapping,
    }


def _build_regions_from_json(
    doc: UnstructuredDocument,
    json_path: str,
    layer_name: str,
    section_config: Dict,
) -> List[StructuredRegion]:
    """
    Build StructuredRegion objects from a claim-extract JSON file.

    Uses section_config to determine which sections to parse and how.

    Page determination uses standard agent-output naming convention:
    - Filename format is "NNNNN_tX.json" where NNNNN is 1-indexed page number
    - Falls back to source.page_index from JSON if filename parsing fails
    """
    json_data = load_json_file(json_path, safe_parse=True)
    if not json_data:
        return []

    # Handle both single-claim format and legacy multi-claim format
    # New format: { "claim_uid": "...", "source": {...}, ... } - single claim object
    # Legacy format: { "claims": [ {...}, {...} ] } - array of claims
    claims = []
    if "claims" in json_data:
        # Legacy multi-claim format - emit warning
        logging.warning(
            f"[DEPRECATED] File {json_path} uses legacy 'claims' array format. "
            "Please update to single-claim format (one claim per file). "
            "This format will be removed in a future version."
        )
        claims = json_data.get("claims", [])
    elif "claim_uid" in json_data or "source" in json_data:
        # New single-claim format - wrap in list for uniform processing
        claims = [json_data]
    else:
        logging.warning(
            f"Unrecognized JSON format in {json_path}: missing 'claims' array or single claim fields"
        )
        return []

    if not claims:
        return []

    # Determine page and table suffix from filename using standard agent-output naming convention
    # Filename format: 00002_t0.json -> page 2 (1-indexed) -> page_index 1 (0-indexed), table suffix t0
    filename = os.path.basename(json_path)
    page_from_filename = None
    table_suffix = "t0"  # Default table suffix
    try:
        parts = filename.replace('.json', '').split('_')
        page_num_str = parts[0]
        page_from_filename = int(page_num_str) - 1  # Convert to 0-indexed
        if len(parts) > 1:
            table_suffix = parts[1]  # e.g., "t0", "t1"
    except (ValueError, IndexError):
        logging.warning(f"Could not parse page number from filename: {filename}")

    regions = []
    json_key_mapping = section_config["json_key_mapping"]
    role_mapping = section_config["role_mapping"]

    for claim in claims:
        claim_uid = claim.get("claim_uid", "unknown")
        source = claim.get("source", {})
        # Use filename-derived page as primary, fall back to JSON page_index
        page_index = (
            page_from_filename
            if page_from_filename is not None
            else source.get("page_index", 0)
        )
        ocr_line_range = source.get("ocr_line_range", [0, 0])

        if len(ocr_line_range) < 2:
            logging.warning(f"Invalid ocr_line_range for claim {claim_uid}")
            continue

        start_line = ocr_line_range[0]
        end_line = ocr_line_range[1]

        # Build region data structure for JsonRegionParser
        # Map claim JSON keys to section titles using config
        region_data = {}

        for json_key, value in claim.items():
            if json_key in ("claim_uid", "source"):
                continue  # Skip metadata fields

            # Try to find matching section title from config
            section_title = json_key_mapping.get(json_key.lower())
            if section_title:
                region_data[section_title] = value
            else:
                # Fallback: use JSON key as section title (uppercase)
                region_data[json_key.upper().replace("_", " ")] = value

        if not region_data:
            logging.debug(f"Claim {claim_uid} has no extractable sections")
            continue

        try:
            # Create parser configured with section types from config
            parser = JsonRegionParser(
                kv_section_titles=section_config["kv_section_titles"],
                table_section_titles=section_config["table_section_titles"],
            )

            # Generate unique region ID from page, table suffix, and line range
            # e.g., p1_t0_L37-66 for page 1, table 0, lines 37-66
            region_id = f"p{page_index}_{table_suffix}_L{start_line}-{end_line}"

            # Build the region using the parser
            region = parser.build_single_page_region(
                json_data=region_data,
                region_id=region_id,
                page=page_index,
                page_y=start_line,
                page_h=end_line - start_line,
            )

            # Add required tags for traceability
            region.tags["source"] = "region-builder"
            region.tags["source_layer"] = layer_name
            region.tags["claim_uid"] = claim_uid
            region.tags["table_suffix"] = table_suffix

            # Add role_hint tags to sections from config
            for section in region.sections_flat():
                title_upper = (section.title or "").upper()
                role = role_mapping.get(title_upper)
                if role:
                    section.tags["role_hint"] = role

            regions.append(region)
            logging.debug(f"Created region for claim {claim_uid} on page {page_index}")

        except Exception as e:
            logging.error(f"Failed to build region for claim {claim_uid}: {e}")
            continue

    return regions
