import re
import uuid
from collections import deque
from typing import Any, Dict, List, Union

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.engine.transform import transform_field_value
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    Field,
    MatchFieldRow,
    MatchSection,
    MatchSectionType,
)
from marie.extract.models.span import Span
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import TableBlock, TableSeries
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

        # This is the list of rules that defines how to parse each section role.
        parser_sections_rules = region_parser_cfg.get("sections", [])

        # Find all `StructuredRegion` objects that are within the scope of the MatchSection's spans.
        pages_in_section = sorted(list(set(s.page for s in section.span)))
        if not pages_in_section:
            self.logger.info("Section has no page spans; skipping region processing.")
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

                if not section_rule:
                    raise ValueError(
                        "No rule for this role_hint, so we can't process it."
                    )
                    # continue  # No rule for this role_hint, so we can't process it.

                parse_method = section_rule.get("parse")
                self.logger.info(
                    f"Found structured section '{structured_section.title}' with role_hint '{role_hint}'. Parsing as '{parse_method}'."
                )

                # Delegate to the appropriate processor based on the parse method.
                if parse_method == "table":
                    self.logger.info(
                        f"Table processing for region with role_hint '{role_hint}' is not yet implemented."
                    )
                    self._process_region_as_table(
                        regions_cfg,
                        section,  # The original MatchSection to populate with results
                        structured_section,
                        template_fields_repeating,
                    )
                elif parse_method == "kv":
                    self.logger.info(
                        f"KV processing for region with role_hint '{role_hint}' is not yet implemented."
                    )
                else:
                    self.logger.warning(
                        f"Unsupported parse method '{parse_method}' for role_hint '{role_hint}'."
                    )

        #
        # table_cfg = region_entry.get("table", {}) or {}
        # body_cfg = table_cfg.get("body", {}) or {}
        # footer_cfg = table_cfg.get("footer", {}) or {}
        #
        # # Build field maps from the region's table configuration
        # field_to_header_map: Dict[str, Dict[str, object]] = {}
        # field_to_footer_map: Dict[str, Dict[str, object]] = {}
        #
        # columns_cfg = body_cfg.get("columns", {}) or {}
        # for field_name, field_info in columns_cfg.items():
        #     field_to_header_map[field_name] = {
        #         "selectors": field_info.get("annotation_selectors", []),
        #         "primary": field_info.get("primary", False),
        #         "level": "SERVICE_LINE",
        #     }
        #
        # # FIXME/TODO: Support footer config properly
        # footer_columns_cfg = (
        #     (footer_cfg.get("columns") or {}) if isinstance(footer_cfg, dict) else {}
        # )
        # for field_name, field_info in footer_columns_cfg.items():
        #     field_to_footer_map[field_name] = {
        #         "selectors": field_info.get("annotation_selectors", []),
        #         "level": "DOCUMENT",
        #     }
        #
        # # Access field mappings prepared on the layer (used for value transforms)
        # self.logger.info(f"Processing regions (table) for layer: {layer.layer_name}")
        # self.logger.debug(f"field_to_header_map: {field_to_header_map}")
        # self.logger.debug(f"field_to_footer_map: {field_to_footer_map}")
        #
        # try:
        #     # Collect all tables that fall within the regions spans
        #     pass
        # except Exception as e:
        #     self.logger.error(f"Error processing regions: {e}")
        #     raise e

    def _process_region_as_table(
        self,
        regions_cfg: List[Dict],
        match_section_to_populate: MatchSection,
        structured_section: Any,
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

        # Normalize header text for matching: collapse whitespace and uppercase
        def norm_text(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip()).upper() if isinstance(s, str) else ""

        # Map normalized header text from the config to the field name (config key)
        header_text_to_field_name = {
            norm_text(k): k
            for k in columns_cfg.keys()
            # norm_text(k.replace("_", " ")): k for k in columns_cfg.keys()
        }

        # Process each table block found in the region
        for tbl_idx, table_block in enumerate(table_blocks):
            if not table_block.header_binding:
                continue

            # Map the table's column indices to field names using the header text
            header_map: Dict[int, str] = {
                col_idx: header_text_to_field_name[norm_text(header_cell.text)]
                for col_idx, header_cell in enumerate(table_block.header_binding)
                if norm_text(header_cell.text) in header_text_to_field_name
            }

            if not header_map:
                self.logger.warning(
                    f"Could not map any headers for table in '{structured_section.title}'. Headers: {[h.text for h in table_block.header_binding]}"
                )
                continue

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
        field_mappings = layer.non_repeating_field_mappings

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

        self.logger.info(f"Processing layer: {layer.layer_name}")
        extracted_fields = []

        for span in spans:
            self.logger.info(f"Processing span: {span}")
            page_id = span.page
            lines = document.lines_for_page(page_id)
            # Determine line range based on span location
            start_line = span.y
            end_line = start_line + span.h

            plucked_lines = [
                line for line in lines if start_line <= line.metadata.line_id < end_line
            ]

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
