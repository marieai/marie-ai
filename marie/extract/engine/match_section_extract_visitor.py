import re
import uuid
from collections import deque
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.engine.transform import transform_field_value
from marie.extract.models.definition import FieldMapping, FieldScope
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    Field,
    MatchFieldRow,
    MatchSection,
    MatchSectionType,
)
from marie.extract.models.span import Span
from marie.extract.results.span_util import pluck_lines_by_span
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import (
    KVList,
    RegionPart,
    RowRole,
    Section,
    SectionRole,
    StructuredRegion,
    TableBlock,
    TableRow,
    TableSeries,
)
from marie.extract.structures.table import Table
from marie.logging_core.logger import MarieLogger


def stringify(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    collapsed_string = re.sub(r'\s+', ' ', value).strip()
    return collapsed_string


class MatchSectionExtractionProcessingVisitor(BaseProcessingVisitor):
    """
    Extract values from the matched sections.
    """

    def __init__(self, enabled: bool):
        super().__init__(enabled)
        self.logger = MarieLogger(context=self.__class__.__name__)
        # TODO : Add dynamic engine loading and extraction

    def visit(self, context: ExecutionContext, parent: MatchSection) -> None:
        self.logger.info("----------------------------------------")
        self.logger.info("Processing MatchSectionExtractionProcessingVisitor")
        queue = deque([parent])
        while queue:
            current = queue.popleft()
            if current is None:
                continue
            self.logger.info(f'---- Extracting from : {current.type}')
            if current.type == MatchSectionType.CONTENT:
                self.process_section(context, parent, current)
            queue.extend(current.sections)
        self.logger.info("Finished processing MatchSectionExtractionProcessingVisitor")
        self.logger.info("----------------------------------------")

    def process_section(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        """
        Processes a given section within a document layer to extract field values
        based on defined selectors and annotations.

        Args:
            context (ExecutionContext): The execution context containing the document.
            parent (MatchSection): The parent section in the document hierarchy.
            section (MatchSection): The current section to process.
        """
        assert context is not None, "Execution context must not be None."
        assert section is not None, "Section must not be None."
        assert parent is not None, "Parent section must not be None."
        assert (
            section.owner_layer is not None
        ), "Section must be associated with a layer."
        assert context.document is not None, "Context must include a document."

        self.process_fields(context, parent, section)
        self.process_regions(context, parent, section)
        # self.process_tables(context, parent, section)

    def _ci(self, s: str) -> str:
        return (s or "").casefold()

    def _parse_selector_regex_hint(self, selector: str) -> tuple[str, bool]:
        """
        Parses inline regex hints for a selector, returning (clean_selector, is_regex).
        Accepts:
          - "re:<pattern>"
          - "/<pattern>/"  (slashes must be at both ends)
        """
        sel = selector or ""
        if sel.startswith("re:"):
            return sel[3:], True
        if len(sel) >= 2 and sel[0] == "/" and sel[-1] == "/":
            return sel[1:-1], True
        return sel, False

    def _selector_matches_text(
        self, selector: str, text: str, use_regex_flag: bool
    ) -> bool:
        """
        Case-insensitive match. If use_regex_flag is True or selector has inline regex hint,
        match as regex; otherwise do a literal contains with casefold.
        """

        sel, hinted_regex = self._parse_selector_regex_hint(selector)
        is_regex = use_regex_flag or hinted_regex
        if is_regex:
            try:
                return re.search(sel, text, flags=re.IGNORECASE) is not None
            except re.error:
                # Fallback to literal compare/contains if regex is invalid
                return self._ci(sel) in self._ci(text)
        return self._ci(sel) in self._ci(text)

    def _collect_selectors_from_cfg(self, cfg: dict) -> tuple[list[str], bool]:
        """
        From a column or field config dict, returns (selectors, use_regex_flag).
        Accepts `annotation_selectors`, `selectors`, or `selector` keys.
        """
        selectors = []
        use_regex_flag = False
        if isinstance(cfg, dict):
            if "annotation_selectors" in cfg and isinstance(
                cfg["annotation_selectors"], list
            ):
                selectors = [str(s) for s in cfg["annotation_selectors"] if s]
            elif "selectors" in cfg and isinstance(cfg["selectors"], list):
                selectors = [str(s) for s in cfg["selectors"] if s]
            elif "selector" in cfg and cfg["selector"]:
                selectors = [str(cfg["selector"])]
            use_regex_flag = bool(cfg.get("use_regex", False))
        return selectors, use_regex_flag

    def process_regions(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        """
        Sister method to `process_tables` that operates on the new `regions` configuration.
        Currently supports type: table regions and reuses the same extraction flow by
        building header/footer mappings from the region entry matching the section title.
        """
        self.logger.info("Processing regions section")
        assert context is not None, "Execution context must not be None."
        assert section is not None, "Section must not be None."
        assert parent is not None, "Parent section must not be None."
        assert (
            section.owner_layer is not None
        ), "Section must be associated with a layer."
        assert context.document is not None, "Context must include a document."

        document = context.document
        layer = section.owner_layer
        spans: List[Span] = section.span

        # Regions configuration is expected to be present on the layer (loaded directly from YAML `regions:`)
        region_parser_cfg, regions_cfg, template_fields_repeating = (
            layer.regions_config_raw
        )
        region_parser_cfg = OmegaConf.to_container(region_parser_cfg, resolve=True)
        regions_cfg = OmegaConf.to_container(regions_cfg, resolve=True)
        template_fields_repeating = OmegaConf.to_container(
            template_fields_repeating, resolve=True
        )

        # FIXME :
        #   This is a clusterfuck in the way we handle the config for repeating fields and non-repeating fields;
        #   for non-repeating fields we have field mappings on the layer that contain the field definition and the mapping
        #   and for repeating fields we have the raw config that contains the field definitions only
        #   this are our non-repeating fields that we need to map to KV values if needed
        field_mappings: List[FieldMapping] = layer.non_repeating_field_mappings

        # Unified list of all field mappings for the layer.
        all_field_mappings: List[FieldMapping] = layer.fields

        parser_sections_rules = region_parser_cfg.get("sections", [])
        # Collect all regions fully contained by any of the section spans (line-based)
        regions_in_scope = set()
        if not spans:
            self.logger.info("Section has no spans; skipping region processing.")
            return

        for span in spans:
            page_id = span.page
            start_line = span.y
            end_line = start_line + span.h

            regions_by_page = document.regions_for_page(page_id)
            for region in regions_by_page:
                try:
                    # Compute region line range from its parts' spans on this page
                    mins: List[int] = []
                    maxs: List[int] = []
                    for part in region.parts:
                        ps = part.span
                        if ps.page != page_id:
                            continue
                        mins.append(int(ps.y))
                        maxs.append(int(ps.y + ps.h))

                    if not mins or not maxs:
                        self.logger.warning(
                            f"Region {region} has no parts on page {page_id}; skipping."
                        )
                        continue

                    region_start = min(mins)
                    region_end = max(maxs)
                    # Fully-contained check
                    if region_start > start_line and region_end < end_line:
                        regions_in_scope.add(region)
                except Exception:
                    raise

        if not regions_in_scope:
            self.logger.warning(
                f"No structured regions found within spans for section '{section.label}'"
            )
            return

        # SECOND BASIC METHOD USED FOR TESTING ONLY : Process all regions on the pages covered by the section's spans
        if False:
            parser_sections_rules = region_parser_cfg.get("sections", [])

            # Find all `StructuredRegion` objects that are within the scope of the MatchSection's spans.
            pages_in_section = sorted(list(set(s.page for s in section.span)))
            if not pages_in_section:
                self.logger.info(
                    "Section has no page spans; skipping region processing."
                )
                return

            # Collect all unique StructuredRegions that fall within the section's pages
            regions_in_scope = set()
            for page_id in pages_in_section:
                regions_on_page = document.regions_for_page(page_id)
                for r in regions_on_page:
                    regions_in_scope.add(r)

            if not regions_in_scope:
                self.logger.info(
                    f"No structured regions found on pages {pages_in_section} for section '{section.label}'"
                )
                return

        # Iterate through all sections of all scoped regions and process them based on their `role_hint` tag.
        for region in regions_in_scope:
            for structured_section in region.sections_flat():
                role_hint = structured_section.tags.get("role_hint")
                if not role_hint:
                    self.logger.warning(
                        f'Role hint for section {structured_section.title} not found.'
                    )
                    continue

                # Find the parsing rule for this section's role hint.
                section_rule = next(
                    (
                        rule
                        for rule in parser_sections_rules
                        if rule.get("role") == role_hint
                    ),
                    None,
                )

                # Not every role_hint needs to have a rule; some may be informational only or processed differently aka(like lookup tables)
                # if the section is intereset then it needs to have a rule to be processed even if it is a no-op rule

                if not section_rule:
                    # USED FOR DEBUG ONLY
                    if False:
                        raise ValueError(
                            f"No rule for role_hint `{role_hint}` so we can't process it."
                        )
                    self.logger.warning(
                        f"No rule for role_hint `{role_hint}` so we can't process it."
                    )
                    continue  # No rule for this role_hint, so we can't process it.

                parse_method = section_rule.get("parse")
                self.logger.info(
                    f"Found structured section '{structured_section.title}' with role_hint '{role_hint}'. Parsing as '{parse_method}'."
                )

                # Delegate to the appropriate processor based on the parse method.
                if parse_method == "table":
                    self.logger.info(
                        f"Table processing for region with role_hint '{role_hint}'."
                    )
                    self._process_region_as_table(
                        regions_cfg,
                        section,  # The original MatchSection to populate with results
                        structured_section,
                        template_fields_repeating,
                    )
                elif parse_method == "kv":
                    self.logger.info(
                        f"KV processing for region with role_hint '{role_hint}' ."
                    )

                    self._process_region_as_kv(
                        regions_cfg,
                        section,  # The original MatchSection to populate with results
                        structured_section,
                        field_mappings,
                        all_field_mappings,
                    )

                else:
                    self.logger.warning(
                        f"Unsupported parse method '{parse_method}' for role_hint '{role_hint}'."
                    )

    def _process_region_as_kv(
        self,
        regions_cfg: List[Dict],
        match_section: MatchSection,
        structured_section: Section,
        field_mappings: List[FieldMapping],
        all_field_mappings: List[FieldMapping],
    ) -> None:
        """
        Process a structured section configured as key-value (kv).

        Reads `fields` from the region entry:
            - title: <SECTION TITLE>
              type: kv
              role: <role-name>
              fields:
                FIELD_NAME_A:
                  annotation_selectors: [ "LABEL_A", "LABEL_A_ALT" ]
                FIELD_NAME_B:
                  annotation_selectors: [ "LABEL_B" ]
                ...

        Returns:
            List of created field objects (as produced by create_fields).
        """

        #  Find region entry and validate type
        sec_title_upper = (structured_section.title or "").strip().upper()
        region_entry = next(
            (
                entry
                for entry in regions_cfg
                if str(entry.get("title", "")).strip().upper() == sec_title_upper
            ),
            None,
        )
        if not region_entry or region_entry.get("type") != "kv":
            self.logger.warning(
                f"No 'kv' region config found for section '{structured_section.title}'."
            )
            return

        fields_cfg = region_entry.get("fields", {}) or {}
        if not fields_cfg:
            self.logger.warning(
                f"No 'fields' configured for kv region '{structured_section.title}'. Nothing to extract."
            )
            return

        # TODO: Initial implementation to use role_hints and Scoped Fields
        role_hint = structured_section.tags.get("role_hint")
        # Filter field mappings to only include those relevant for this region's role and scope.
        field_mappings_filtered = [
            fm
            for fm in all_field_mappings
            if fm.scope == FieldScope.REGION and fm.role == role_hint
        ]

        if not field_mappings_filtered:
            self.logger.info(
                f"No REGION-scoped field mappings with role '{role_hint}' found for section '{structured_section.title}'"
            )

        if match_section.fields is None:
            match_section.fields = []

        # Build field -> (selectors, use_regex_flag)
        kv_specs: Dict[str, tuple[list[str], bool]] = {}
        for field_name, field_info in fields_cfg.items():
            selectors, use_regex_flag = self._collect_selectors_from_cfg(field_info)
            if selectors:
                kv_specs[field_name] = (selectors, use_regex_flag)

        if not kv_specs:
            self.logger.info(
                f"No valid selectors for kv region '{structured_section.title}'."
            )
            return

        #  Walk KVList blocks and match selectors against item.key
        populated_fields: set[str] = set()
        template_field_mappings = {}
        extracted_fields = []

        for mapping in field_mappings:
            field_def = mapping.field_def
            template_field_mappings[field_def['name']] = field_def

        for block in structured_section.blocks:
            # Expect KVList-like block: must have `items`
            try:
                if not isinstance(block, KVList):
                    raise TypeError
                items = block.items
            except Exception:
                continue
            if not items:
                continue

            # For quick lookups, prepare a list of (key_text, value_text, item)
            kv_triplets = []
            for it in items:
                try:
                    key_text = it.key or ""
                except Exception:
                    key_text = ""
                try:
                    value_text = it.value or ""
                except Exception:
                    value_text = ""
                kv_triplets.append((key_text, value_text, it))

            # Attempt to populate each configured field once
            for field_name, (selectors, use_regex_flag) in kv_specs.items():
                if field_name in populated_fields:
                    continue

                match_generator = (
                    (value_text, it, sel)
                    for sel in selectors
                    for key_text, value_text, it in kv_triplets
                    if key_text
                    and self._selector_matches_text(sel, key_text, use_regex_flag)
                )

                first_match = next(match_generator, None)

                if not first_match:
                    continue

                matched_value, matched_item, matched_selector = first_match
                value_text = matched_item.value or ""

                self.logger.info(
                    f"Extracting KV field `{field_name}` = '{value_text}' via selector '{matched_selector}' (key='{matched_item.key}')"
                )

                # Resolve field definition
                field_def = template_field_mappings.get(field_name, {}) or {}
                field_def = dict(field_def)  # shallow copy
                field_def["name"] = field_name
                # Optional type override fallback if template missing; many totals are MONEY
                field_def.setdefault("type", "MONEY")

                transformed_value = transform_field_value(field_def, value_text)
                # this is a dummy line_with_meta; we don't have line-level metadata for KV values
                faux_line_with_meta = LineWithMeta(
                    line=value_text,
                    metadata=None,
                    annotations=[],
                )

                created = self.create_fields(
                    field_def, value_text, transformed_value, faux_line_with_meta
                )
                extracted_fields.extend(created)

        # Attach kv fields to the matched section.
        # TODO: we will change this to a dictionary of field types
        if match_section.matched_non_repeating_fields is None:
            match_section.matched_non_repeating_fields = []
        match_section.matched_non_repeating_fields.extend(extracted_fields)

    def _process_region_as_table(
        self,
        regions_cfg: List[Dict],
        match_section_to_populate: MatchSection,
        structured_section: Section,
        template_fields_repeating: Dict,
    ):
        """Helper to process a structured section that contains table data."""
        # Extract all table blocks from the structured section
        table_blocks: List[TableBlock] = []
        for block in structured_section.blocks:
            if isinstance(block, TableBlock):
                table_blocks.append(block)
            elif isinstance(block, TableSeries):
                table_blocks.extend(block.segments)
            else:
                raise TypeError

        if not table_blocks:
            return

        # Find the extraction configuration from the `regions:` block in the YAML.
        # This is matched by the title of the structured section.
        sec_title_upper = (structured_section.title or "").strip().upper()
        region_entry = next(
            (
                entry
                for entry in regions_cfg
                if str(entry.get("title", "")).strip().upper() == sec_title_upper
            ),
            None,
        )

        if not region_entry or region_entry.get("type") != "table":
            self.logger.warning(
                f"No 'table' region config found for section '{structured_section.title}'. Cannot map columns."
            )
            return

        columns_cfg = (
            region_entry.get("table", {}).get("body", {}).get("columns", {}) or {}
        )
        if not columns_cfg:
            self.logger.warning(
                f"No 'columns' configured for table region '{structured_section.title}'."
            )
            return

        # There is only one table config per region name, but each labeled region can have one table block
        table_config = region_entry.get("table", {})
        field_to_header_map = {}
        field_to_footer_map = {}  # FOOTER ARE NOT SUPPORTED YET or MAYBE EVEN EVER

        if 'body' in table_config and 'columns' in table_config['body']:
            for field_name, field_info in table_config['body']['columns'].items():
                field_to_header_map[field_name] = {
                    "selectors": field_info.get('annotation_selectors', []),
                    "primary": field_info.get('primary', False),
                    "level": region_entry[
                        'role'
                    ],  # Default to SERVICE_LINE for table body
                }

        # NOT SUPPORTED YET - PLACEHOLDER
        # Process footer columns
        if False:
            if 'footer' in table_config and 'columns' in table_config['footer']:
                for field_name, field_info in table_config['footer']['columns'].items():
                    field_to_footer_map[field_name] = {
                        "selectors": field_info.get('annotation_selectors', []),
                        "level": "DOCUMENT",  # Footer values are at document level
                    }

        # Process each span in the section
        self.logger.info(f'field_to_header_map: {field_to_header_map}')
        self.logger.info(f'field_to_footer_map: {field_to_footer_map}')

        # Now process each TableBlock
        self.logger.info(
            f"Identified {len(table_blocks)} table block(s) to process in this region"
        )

        for tb in table_blocks:
            if not tb.rows:
                self.logger.warning("TableBlock has no rows; skipping.")
                continue

            # Use RowRole to separate header and body
            header_row = None
            body_rows = []
            for r in tb.rows:
                if r.role == RowRole.HEADER and header_row is None:
                    header_row = r
                elif r.role == RowRole.BODY:
                    body_rows.append(r)

            if header_row is None:
                self.logger.warning(
                    "No header row (RowRole.HEADER) found; skipping table block."
                )
                continue
            if not body_rows:
                self.logger.warning(
                    "No body rows (RowRole.BODY) found; skipping table block."
                )
                continue

            # Derive page id from header row; fallback to first body row
            page_id = (
                header_row.source_page
                if header_row
                else (body_rows[0].source_page if body_rows else -1)
            )
            self.logger.info(f"Processing table block for page: {page_id}")

            # Prefer canonical headers from header_binding; otherwise, fallback to header row cell strings
            if tb.header_binding and len(tb.header_binding) > 0:
                header_texts = list(tb.header_binding)
            else:
                # Fallback: stringify header cells
                # header_texts = [str(c) if c is not None else "" for c in header_row.cells]
                raise NotImplementedError(
                    "Fallback to header row cell text is not implemented yet."
                )

            columns_to_process = {}
            claimed_columns = set()

            for field_name, header_cfg in field_to_header_map.items():
                selectors, use_regex_flag = self._collect_selectors_from_cfg(header_cfg)
                if not selectors:
                    self.logger.warning(
                        f"No selectors defined for field '{field_name}', skipping header match."
                    )
                    continue

                processed_column = -1
                matched = False
                for selector in selectors:
                    for col_index, header_text in enumerate(header_texts):
                        if col_index in claimed_columns or not header_text:
                            continue
                        if self._selector_matches_text(
                            selector, header_text, use_regex_flag
                        ):
                            self.logger.info(
                                f"Matched header '{selector}' for field '{field_name}' at column {col_index} "
                                f"(header='{header_text}')"
                            )
                            processed_column = col_index
                            claimed_columns.add(col_index)
                            matched = True
                            break
                    if matched:
                        break

                if processed_column != -1:
                    columns_to_process[field_name] = {
                        "cell_index": processed_column,
                        "header_config": header_cfg,
                    }
                else:
                    self.logger.debug(
                        f"No header match found for field '{field_name}'. "
                        f"Selectors tried: {selectors}. Headers: {header_texts}"
                    )

            # columns_to_process now maps field_name -> {cell_index, header_cfg}
            # Next step (not shown here): iterate body_rows and use columns_to_process indices to extract values

            # Sort `columns_to_process` by `cell_index` key
            columns_to_process = dict(
                sorted(
                    columns_to_process.items(),
                    key=lambda item: item[1]['cell_index'],
                )
            )

            self.logger.info(f"Columns to process mapping: {columns_to_process}")

            # Extract rows using the resolved column indices
            matched_field_rows: List[MatchFieldRow] = self._build_matched_field_rows(
                body_rows=body_rows,
                columns_to_process=columns_to_process,
                page_id=page_id,
                template_fields_repeating=template_fields_repeating,
            )

            match_section_to_populate.matched_field_rows = matched_field_rows

    def _build_matched_field_rows(
        self,
        body_rows: List[TableRow],
        columns_to_process: Dict[str, Dict[str, Any]],
        page_id: int,
        template_fields_repeating: Dict[str, Any],
    ) -> List[MatchFieldRow]:
        """
        Build MatchFieldRow list by extracting values from body rows using resolved column indices.

        Parameters:
            body_rows: list of TableRow objects with role BODY
            columns_to_process: mapping of field name -> { cell_index: int, header_config: dict }
            page_id: page identifier to propagate to line metadata
            template_fields_repeating: field configuration template for repeating fields

        Returns:
            List[MatchFieldRow]
        """
        matched_field_rows: List[MatchFieldRow] = []

        if not body_rows or not columns_to_process:
            return matched_field_rows

        # Stable processing order
        ordered_fields = [
            k
            for k, _ in sorted(
                columns_to_process.items(),
                key=lambda item: item[1]["cell_index"],
            )
        ]

        for row in body_rows:
            extracted_cells = []
            self.logger.info("row : *******************")

            cells = row.cells
            for field_name in ordered_fields:
                column_def = columns_to_process[field_name]
                column_index = int(column_def["cell_index"])
                header_config = column_def["header_config"]

                if column_index < 0 or column_index >= len(cells):
                    self.logger.debug(
                        f"Column index {column_index} out of range for row; skipping field '{field_name}'."
                    )
                    continue

                cell = cells[column_index]

                # Prefer the first line text if available: we want the LLM to aggregate the cell lines
                if cell.lines and len(cell.lines) > 0:
                    root_line = cell.lines[0]
                    if root_line.metadata:
                        root_line.metadata.page_id = page_id  # FIXME: consider removing once not needed downstream
                    cell_value = root_line.line or ""
                else:
                    root_line = None
                    cell_value = str(cell) if cell is not None else ""

                self.logger.debug(
                    f"Extracting value for `{field_name}` = '{cell_value}' from column index {column_index}"
                )

                # Copy field definition to avoid mutating the template
                field_def = dict(template_fields_repeating.get(field_name, {}) or {})
                field_def["name"] = field_name

                transformed_value: Union[str, float, dict[str, None]] = (
                    transform_field_value(field_def, cell_value)
                )
                self.logger.info(f"transformed_value : {transformed_value}")

                fields = self.create_fields(
                    field_def, cell_value, transformed_value, root_line
                )
                extracted_cells.extend(fields)

            matched_field_row: MatchFieldRow = MatchFieldRow(fields=extracted_cells)
            matched_field_rows.append(matched_field_row)

        return matched_field_rows

    def process_tables(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        """
        Processes a given section within a document layer to extract table values
        based on defined selectors and annotations.

        Args:
            context (ExecutionContext): The execution context containing the document.
            parent (MatchSection): The parent section in the document hierarchy.
            section (MatchSection): The current section to process.
        """
        # Extract table configuration from YAML

        document = context.document
        layer = section.owner_layer
        spans: List[Span] = section.span
        # table_configs = layer.table_config_raw
        table_configs, template_fields_repeating = (
            layer.table_config_raw
        )  # TODO: this needs to be converted to a concrete object
        field_mappings: List[FieldMapping] = layer.non_repeating_field_mappings

        self.logger.info(f"Processing layer: {layer.layer_name}")
        # Build field to header mapping (instead of header to field)
        field_to_header_map = {}
        field_to_footer_map = {}

        for table_name, table_config in table_configs.items():
            if 'body' in table_config and 'columns' in table_config['body']:
                for field_name, field_info in table_config['body']['columns'].items():
                    field_to_header_map[field_name] = {
                        "selectors": field_info.get('annotation_selectors', []),
                        "primary": field_info.get('primary', False),
                        "level": "SERVICE_LINE",  # Default to SERVICE_LINE for table body
                    }

            # Process footer columns
            if 'footer' in table_config and 'columns' in table_config['footer']:
                for field_name, field_info in table_config['footer']['columns'].items():
                    field_to_footer_map[field_name] = {
                        "selectors": field_info.get('annotation_selectors', []),
                        "level": "DOCUMENT",  # Footer values are at document level
                    }

        # Process each span in the section
        print('field_to_header_map:', field_to_header_map)
        print('footer_field_map:', field_to_footer_map)
        try:
            # collect all the tables for each span
            tables = []
            for span in spans:
                print(f'span : {span}')

                page_id = span.page
                tables_by_page: List[Table] = document.tables_for_page(page_id)
                start_line = span.y
                end_line = start_line + span.h

                for table in tables_by_page:
                    rows = table.cells
                    table_meta = table.metadata
                    table_line_id = table_meta.line.metadata.line_id
                    table_max_line_id = table_line_id + len(rows)

                    if table_line_id > start_line and table_max_line_id < end_line:
                        # Check if the table is within the span's line range
                        if start_line <= table_line_id < end_line:
                            self.logger.info(f"Table found within span: {table}")
                            tables.append(table)

            self.logger.info(
                f"Collected tables for section {len(tables)} : '{section.label}'"
            )
            # TODO:
            # If multiple tables are found we need to have a way to process them independently
            # How will we handle multiple tables in a section? Are they all related?

            if len(tables) > 1:
                self.logger.warning(
                    f"Multiple tables found for section '{section.label}'. Only the first one will be processed."
                )

            # now we have all the tables for this section to process
            for table in tables:
                page_id = table.metadata.page_id
                self.logger.info(f"Processing table for page: {page_id}")

                rows = table.cells
                header_row = rows[0]
                # TODO : Currently we match on teh CELL TEXT AS THE HEADER to SELECTORS
                # I like to have a better way to match the header with the selector down the line
                columns_to_process = {}
                for field_name, header_config in field_to_header_map.items():
                    processed_column = -1
                    if 'selectors' in header_config:
                        for selector in header_config['selectors']:
                            # Check if the selector is present in the header row
                            for cell_index, cell in enumerate(header_row):
                                if selector in cell.lines[0].line:
                                    self.logger.info(
                                        f"Matched header '{selector}' for field '{field_name}'"
                                    )
                                    processed_column = cell_index
                                    break

                    if processed_column != -1:
                        columns_to_process[field_name] = {
                            "cell_index": processed_column,
                            "header_config": header_config,
                        }

                self.logger.info(f"Columns to process mapping: {columns_to_process}")
                # TODO: Add footer detection logic from annotations as primary, and fallback to field_match if needed

                footer_config = table_config.get("footer", {}).get("detect_by", {})
                has_footer = False
                footer_row = None

                if "field_match" in footer_config:
                    field_match_criteria = footer_config["field_match"]
                    match_type = footer_config.get(
                        "match_type", "all"
                    )  # Default to 'all' if not specified

                    for row_index, row in enumerate(rows):
                        match_count = 0
                        for match_criteria in field_match_criteria:
                            column_name = match_criteria.get("column")
                            pattern = match_criteria.get("pattern")

                            if column_name and pattern:
                                for column_index, cell in enumerate(row):
                                    if column_index < len(header_row):
                                        header_cell = (
                                            header_row[column_index].lines[0].line
                                            if (header_row[column_index].lines)
                                            else ""
                                        )
                                        if header_cell == column_name and cell.lines:
                                            cell_value = cell.lines[
                                                0
                                            ].line.strip()  # Strip extra whitespace
                                            if re.match(
                                                pattern, cell_value, re.IGNORECASE
                                            ):  # Add case-insensitive match
                                                match_count += 1
                                                break

                        # Determine match based on 'any' or 'all' logic
                        if (
                            match_type == "all"
                            and match_count == len(field_match_criteria)
                        ) or (match_type == "any" and match_count > 0):
                            self.logger.info(
                                f"Footer row detected based on criteria: {field_match_criteria} with match_type: {match_type}."
                            )
                            has_footer = True
                            footer_row = row
                            break

                    # FIXME : THIS IS A HACK just for one template to work
                    # this has to be an expression that we call call dynamically
                    # Check if first two cells are empty
                    if not has_footer:
                        row = rows[-1]
                        if (
                            len(row) >= 2
                            and (not row[0].lines or not row[0].lines[0].line.strip())
                            and (not row[1].lines or not row[1].lines[0].line.strip())
                        ):
                            has_footer = True
                            footer_row = row
                            self.logger.info(
                                "Footer row detected based on empty first two cells."
                            )

                # Fallback for flexible footer detection
                flexible_match_config = footer_config.get("flexible_match", {})
                if not has_footer and flexible_match_config.get("enabled", False):
                    fallback_pattern = flexible_match_config.get(
                        "pattern", "TOTAL:?$"
                    )  # Default fallback pattern

                    for row in rows:
                        for cell in row:
                            if cell.lines:
                                cell_value = (
                                    cell.lines[0].line.strip().upper()
                                )  # Convert to uppercase for consistent matching

                                print(
                                    f'fallback_pattern : {fallback_pattern} >>> {cell_value}'
                                )
                                if re.search(
                                    fallback_pattern, cell_value, re.IGNORECASE
                                ):  # Match against regex
                                    self.logger.info(
                                        f"Footer row detected using flexible match with pattern: {fallback_pattern}."
                                    )
                                    has_footer = True
                                    footer_row = row
                                    break
                        if has_footer:
                            break

                # If a footer is still not found but always_present is True
                if not has_footer and footer_config.get("always_present", False):
                    self.logger.info(
                        "Footer row detected (default fallback: always present)."
                    )
                    has_footer = True
                    footer_row = rows[-1]  # Assume last row as footer if unspecified

                self.logger.info("Footer detection results:")
                self.logger.info(f"has_footer: {has_footer}")
                self.logger.info(f"footer_row: {footer_row}")

                # Data rows (exclude header and footer if present)
                data_rows = rows[1:-1] if has_footer else rows[1:]  # Skip header row

                # Sort `columns_to_process` by `cell_index` key
                columns_to_process = dict(
                    sorted(
                        columns_to_process.items(),
                        key=lambda item: item[1]['cell_index'],
                    )
                )
                matched_field_rows: List[MatchFieldRow] = []

                for row in data_rows:
                    extracted_cells = []
                    self.logger.info("row : *******************")
                    for field_name, column_def in columns_to_process.items():
                        column_index = int(column_def['cell_index'])
                        header_config = column_def['header_config']
                        cell = row[column_index]
                        # Extract the value from the cell
                        root_line = cell.lines[0]
                        root_line.metadata.page_id = page_id  # FIXME : THIS IS A HACK

                        cell_value = root_line.line if cell.lines else ""
                        self.logger.debug(
                            f"Extracting value for `{field_name}` =  '{cell_value}' from column index {column_index}"
                        )
                        field_def = template_fields_repeating.get(field_name, None)
                        field_def['name'] = field_name
                        transformed_value: Union[str | float | dict[str, None]] = (
                            transform_field_value(field_def, cell_value)
                        )
                        self.logger.debug(f'transformed_value : {transformed_value}')
                        fields = self.create_fields(
                            field_def, cell_value, transformed_value, root_line
                        )
                        extracted_cells.extend(fields)

                    matched_field_row: MatchFieldRow = MatchFieldRow(
                        fields=extracted_cells
                    )
                    matched_field_rows.append(matched_field_row)

                section.matched_field_rows = matched_field_rows

                # Footer Row Processing
                if has_footer and footer_row:
                    self.logger.info(f"Processing footer row: {footer_row}")

                    extracted_footer_fields = []
                    for field_name, footer_def in field_to_footer_map.items():
                        self.logger.info(f"Processing footer field: {field_name}")
                        selectors = footer_def.get("selectors", [])
                        matched_column_index = None

                        # Try matching selector in header to find column index
                        for selector in selectors:
                            for idx, header_cell in enumerate(header_row):
                                if (
                                    header_cell.lines
                                    and selector in header_cell.lines[0].line
                                ):
                                    matched_column_index = idx
                                    break
                            if matched_column_index is not None:
                                break

                        if matched_column_index is not None:
                            cell = footer_row[matched_column_index]
                            cell_value = cell.lines[0].line if cell.lines else ""
                            self.logger.info(
                                f"Extracting footer field `{field_name}` = '{cell_value}' from column {matched_column_index}"
                            )

                            template_field_mappings = {}
                            for mapping in field_mappings:
                                field_def = mapping.field_def
                                template_field_mappings[field_def['name']] = field_def
                            print('template_field_mappings:', template_field_mappings)
                            # FIXME: THis is a hack to get the field def
                            field_def = template_field_mappings.get(field_name, {})
                            field_def['name'] = field_name
                            field_def['type'] = 'MONEY'

                            transformed_value = transform_field_value(
                                field_def, cell_value
                            )

                            footer_field = Field(
                                field_name=field_name,
                                field_type=field_def.get("type"),
                                is_required=False,
                                value=stringify(transformed_value),
                                value_original=stringify(cell_value),
                                composite_field=False,
                                x=0,
                                y=0,
                                width=0,
                                height=0,
                                date_format=field_def.get("date_format"),
                                name_format=field_def.get("name_format"),
                                column_name=field_def.get("column_name"),
                                page=page_id,
                                xdpi=300,
                                ydpi=300,
                                confidence=1,
                                scrubbed=True,
                                uuid=None,
                                reference_uuid=None,
                                layer_name="main-layer",
                            )
                            extracted_footer_fields.append(footer_field)

                    # Attach footer fields to the matched section.
                    # TODO: we will change this to a dictionary of field types
                    if section.matched_non_repeating_fields is None:
                        section.matched_non_repeating_fields = []
                    section.matched_non_repeating_fields.extend(extracted_footer_fields)

                break  # TODO : remove this break to process all tables : how to handle multiple tables in a section?
        except Exception as e:
            self.logger.error(f"Error processing tables: {e}")
            raise e

    def process_fields(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        """
        Processes a given section within a document layer to extract field values
        based on defined selectors and annotations.

        Args:
            context (ExecutionContext): The execution context containing the document.
            parent (MatchSection): The parent section in the document hierarchy.
            section (MatchSection): The current section to process.
        """
        assert context is not None, "Execution context must not be None."
        assert section is not None, "Section must not be None."
        assert parent is not None, "Parent section must not be None."
        assert (
            section.owner_layer is not None
        ), "Section must be associated with a layer."
        assert context.document is not None, "Context must include a document."

        document = context.document
        layer = section.owner_layer
        field_mappings = layer.non_repeating_field_mappings
        spans: List[Span] = section.span
        extracted_fields = []
        self.logger.info(f"Processing layer: {layer.layer_name}")

        # Filter for fields that are defined at the LAYER scope.
        field_mappings_filtered = [
            fm for fm in layer.fields if fm.scope == FieldScope.LAYER
        ]

        if not field_mappings_filtered:
            self.logger.info("No layer-level fields to process.")

        for span in spans:
            self.logger.info(f"Processing span: {span}")
            plucked_lines = pluck_lines_by_span(document, span)

            for line in plucked_lines:
                annotations = line.annotations
                if not annotations:
                    continue

                for mapping in field_mappings:
                    field_def = mapping.field_def
                    selector_set = mapping.selector_set

                    if selector_set is None:
                        self.logger.warning(
                            f"Missing selector set for field mapping: {mapping}"
                        )
                        continue

                    for selector in selector_set.selectors:
                        if selector.strategy != "ANNOTATION":
                            self.logger.warning(
                                f"Unsupported selector strategy: {selector.strategy}"
                            )
                            continue

                        for annotation in annotations:
                            if not isinstance(annotation, TypedAnnotation):
                                raise ValueError(
                                    f"Unknown annotation type: {annotation}"
                                )

                            if selector.text == annotation.name:
                                self.logger.debug(
                                    f"Matched annotation '{annotation.name}' for selector '{selector.text}' "
                                    f"in line: '{line.line or '[No text]'}'"
                                )

                                field_def['name'] = mapping.name
                                transformed_value: Union[
                                    str | float | dict[str, None]
                                ] = transform_field_value(field_def, annotation.value)

                                fields = self.create_fields(
                                    field_def, annotation.value, transformed_value, line
                                )

                                for field in fields:
                                    extracted_fields.append(field)

        section.matched_non_repeating_fields = extracted_fields
        self.logger.info(f"Extracted match fields for section '{section.label}':")
        for field in extracted_fields:
            self.logger.info(f"  -  {field}")

    def create_fields(
        self,
        field_def: dict[str, Any],
        value: str,
        transformed_value: Union[str | float | dict[str, None]],
        line: LineWithMeta,
    ) -> List[Field]:

        page_id = line.metadata.page_id
        field_name = field_def.get("name")
        derived_fields = (
            field_def.get("derived_fields", None)
            if field_def.get("derived_fields", None)
            else None
        )
        # FIXME : This is a hack
        if derived_fields and not isinstance(transformed_value, Dict):
            transformed_value = {
                key: None for key in derived_fields.keys()
            }  # THIS IS JUST TO GET SOME DATA LOADED
            # raise ValueError(f'We have derived config, but transformed_value is not a dict : {transformed_value}')

        composite_field = False
        reference_uuid = None
        fields = []
        src_value = transformed_value

        if derived_fields:
            composite_field = True
            src_value = value
            reference_uuid = uuid.uuid4()  # annotation.reference_uuid

        field = Field(
            field_name=field_name,
            field_type=field_def.get("type"),
            is_required=False,
            value=stringify(src_value),
            value_original=stringify(value),
            composite_field=composite_field,
            x=0,
            y=0,
            width=0,
            height=0,
            date_format=field_def.get("date_format"),
            name_format=field_def.get("name_format"),
            column_name=field_def.get("column_name"),
            page=page_id,
            xdpi=300,
            ydpi=300,
            confidence=1,
            scrubbed=True,  # field_def.get("scrubbed", False),
            uuid=reference_uuid,  # annotation.uuid,
            reference_uuid=None,  # annotation.reference_uuid,
            layer_name="main-layer",
        )
        fields.append(field)

        if derived_fields:
            for derived_key, derived_value in derived_fields.items():
                map_value = transformed_value.get(derived_key, None)
                derived_field = {}
                date_format = None
                name_format = None

                field = Field(
                    field_name=derived_value,
                    field_type=derived_field.get("type"),
                    is_required=False,
                    value=stringify(map_value),
                    value_original=None,
                    composite_field=False,
                    x=0,
                    y=0,
                    width=0,
                    height=0,
                    date_format=date_format,
                    name_format=name_format,
                    column_name=derived_value,
                    page=page_id,
                    xdpi=300,
                    ydpi=300,
                    confidence=1,
                    scrubbed=True,
                    uuid=None,
                    reference_uuid=reference_uuid,
                    layer_name="main-layer",
                )
                fields.append(field)

        return fields
