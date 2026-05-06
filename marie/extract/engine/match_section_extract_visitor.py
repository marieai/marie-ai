import importlib
import json
import re
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.engine.transform import TransformReturnType, transform_field_value
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
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import (
    KVList,
    RowRole,
    Section,
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
        self.logger.debug("----------------------------------------")
        self.logger.debug("Processing MatchSectionExtractionProcessingVisitor")
        queue = deque([parent])
        while queue:
            current = queue.popleft()
            if current is None:
                continue
            self.logger.debug(f'---- Extracting from : {current.type}')
            if current.type == MatchSectionType.CONTENT:
                self.process_section(context, parent, current)
            queue.extend(current.sections)
        self.logger.debug("Finished processing MatchSectionExtractionProcessingVisitor")
        self.logger.debug("----------------------------------------")

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

        # Skip record-backed sections — handled by RecordBackedMatchSectionPopulationVisitor
        if section.tags.get("match_section_source_strategy") == "record_backed":
            return

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
        self.logger.debug("Processing regions section")
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
        region_scoping = region_parser_cfg.get("region_scoping", "strict")
        # Collect all regions fully contained by any of the section spans (line-based)
        regions_in_scope = set()
        if not spans:
            self.logger.info("Section has no spans; skipping region processing.")
            return

        # DEBUG: Log section and span info
        self.logger.debug(
            f"=== SCOPING DEBUG: Processing MatchSection '{section.label}' ==="
        )
        self.logger.debug(
            f"  Section spans: {[(s.page, s.y, s.h, s.start_line_id, s.end_line_id) for s in spans]}"
        )

        for span in spans:
            page_id = span.page
            start_line = span.start_line_id
            end_line = span.end_line_id
            # Make sure to include the last line only if our span extends to the end of the page
            if end_line == len(document.lines_for_page(page_id)):
                end_line += 1

            # DEBUG: Log span details
            self.logger.debug(
                f"  Checking page {page_id}: start_line={start_line}, end_line={end_line}"
            )

            regions_by_page = document.regions_for_page(page_id)
            self.logger.debug(
                f"  Found {len(regions_by_page)} regions on page {page_id}"
            )

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

                    # Determine if region is in scope based on scoping strategy
                    if region_scoping == "relaxed":
                        # Majority overlap: region is in scope if >50% of its lines fall within the span
                        overlap_start = max(region_start, start_line)
                        overlap_end = min(region_end, end_line)
                        overlap_length = max(0, overlap_end - overlap_start)
                        region_length = region_end - region_start
                        overlap_ratio = (
                            overlap_length / region_length if region_length > 0 else 0.0
                        )
                        is_in_scope = overlap_ratio >= 0.5
                        self.logger.debug(
                            f"    Region '{region.region_id}': lines {region_start}-{region_end}, "
                            f"scoping=relaxed, overlap={overlap_length}/{region_length} ({overlap_ratio:.1%}), "
                            f"in_scope={is_in_scope}"
                        )
                    else:
                        # Strict: fully contained
                        is_in_scope = (
                            region_start >= start_line and region_end <= end_line
                        )
                        self.logger.debug(
                            f"    Region '{region.region_id}': lines {region_start}-{region_end}, "
                            f"check: {region_start} >= {start_line} AND {region_end} <= {end_line} = {is_in_scope}"
                        )

                    if is_in_scope:
                        regions_in_scope.add(region)
                except Exception:
                    raise

        # Sort regions by page and start line to ensure correct processing order
        # This is critical for maintaining document order when merging sections across regions
        def region_sort_key(region):
            """Sort key: (page, start_line, region_id) based on region's first part span.

            The region_id is used as a tie-breaker when multiple regions have the
            same (page, start_line). This ensures consistent ordering across runs.
            """
            if not region.parts:
                return (float('inf'), float('inf'), "")
            first_part = region.parts[0]
            return (first_part.span.page, first_part.span.y, region.region_id)

        sorted_regions = sorted(regions_in_scope, key=region_sort_key)

        # DEBUG: Log collected regions (now sorted)
        self.logger.debug(
            f"  RESULT: Collected {len(sorted_regions)} regions in scope (sorted): {[r.region_id for r in sorted_regions]}"
        )

        if not sorted_regions:
            self.logger.debug(
                f"No structured regions found within spans for section '{section.label}'"
            )
            return

        # When multiple regions fall within a single MatchSection, each region
        # represents an independent record.  Split the parent into per-region
        # child MatchSections so that each record is processed and rendered
        # separately.  Children receive spans derived from RegionPart.span
        # (which carries the exact OCR line ranges from the source JSON), so
        # the normal BFS path (process_fields → process_regions) handles them
        # without any special flags or guards.
        #
        # Guard: only split sections that were NOT already created by a
        # previous split.  Without this, a child whose spans still overlap
        # a large region (e.g. remarks) would re-split infinitely.
        already_split = section.tags.get("_split_child", False)
        if len(sorted_regions) > 1 and not already_split:
            self._split_multi_region_section(
                context=context,
                document=document,
                section=section,
                sorted_regions=sorted_regions,
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

        # Collect all structured sections from all regions and group by role_hint.
        # This merges sections across regions (and pages) that belong to the same MatchSection.
        # We iterate over sorted_regions to maintain document order (by page, then by start line).
        sections_by_role: Dict[str, List] = {}
        for region in sorted_regions:
            for structured_section in region.sections_flat():
                role_hint = structured_section.tags.get("role_hint")
                if not role_hint:
                    self.logger.warning(
                        f'Role hint for section {structured_section.title} not found.'
                    )
                    continue
                if role_hint not in sections_by_role:
                    sections_by_role[role_hint] = []
                sections_by_role[role_hint].append(structured_section)

        self.logger.debug(
            f"Collected sections by role: {list(sections_by_role.keys())} from {len(sorted_regions)} regions"
        )

        # DEBUG: Dump merged sections as JSON for review
        for role_hint, structured_sections in sections_by_role.items():
            self.logger.debug(
                f"=== DEBUG MERGED SECTIONS JSON for role_hint '{role_hint}' ==="
            )
            for idx, ss in enumerate(structured_sections):
                section_dump = {
                    "index": idx,
                    "title": ss.title,
                    "role": str(ss.role) if ss.role else None,
                    "tags": dict(ss.tags) if ss.tags else {},
                    "blocks": [],
                }
                for block in ss.blocks:
                    if isinstance(block, KVList):
                        block_dump = {
                            "type": "KVList",
                            "items": [
                                {"key": kv.key, "value": kv.value} for kv in block.items
                            ],
                        }
                    elif isinstance(block, TableBlock):
                        block_dump = {
                            "type": "TableBlock",
                            "rows_count": len(block.rows) if block.rows else 0,
                            "header_binding": (
                                list(block.header_binding)
                                if block.header_binding
                                else None
                            ),
                        }
                    elif isinstance(block, TableSeries):
                        block_dump = {
                            "type": "TableSeries",
                            "segments_count": (
                                len(block.segments) if block.segments else 0
                            ),
                        }
                    else:
                        block_dump = {"type": type(block).__name__}
                    section_dump["blocks"].append(block_dump)
                self.logger.debug(
                    f"Section[{idx}]: {json.dumps(section_dump, indent=2)}"
                )

        # Process each role_hint group (merged across all regions in scope)
        for role_hint, structured_sections in sections_by_role.items():
            # Find the parsing rule for this role hint.
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
                        f"No rule for role_hint `{role_hint}` so we can't process it on layer `{layer.layer_name}`."
                    )
                    self.logger.info(
                        f"No rule for role_hint `{role_hint}` so we can't process it on layer `{layer.layer_name}`."
                    )
                continue

            parse_method = section_rule.get("parse")
            self.logger.debug(
                f"Processing {len(structured_sections)} merged sections with role_hint '{role_hint}'. Parsing as '{parse_method}'."
            )

            # Track populated fields across all sections in this role_hint group
            # This prevents duplicate field extraction when multiple regions have the same KV fields
            kv_populated_fields: set[str] = set()

            # Process all sections of this role together
            for structured_section in structured_sections:
                # Delegate to the appropriate processor based on the parse method.
                if parse_method == "table":
                    self.logger.debug(
                        f"Table processing for region with role_hint '{role_hint}'."
                    )
                    self._process_region_as_table(
                        document,
                        regions_cfg,
                        section,  # The original MatchSection to populate with results
                        structured_section,
                        template_fields_repeating,
                    )
                elif parse_method == "kv":
                    self.logger.debug(
                        f"KV processing for region with role_hint '{role_hint}' ."
                    )

                    self._process_region_as_kv(
                        context,  # Pass execution context for field config lookup
                        regions_cfg,
                        section,  # The original MatchSection to populate with results
                        structured_section,
                        field_mappings,
                        all_field_mappings,
                        populated_fields=kv_populated_fields,  # Pass shared set to prevent duplicates
                    )

                else:
                    self.logger.warning(
                        f"Unsupported parse method '{parse_method}' for role_hint '{role_hint}'."
                    )

    def _split_multi_region_section(
        self,
        context: ExecutionContext,
        document,
        section: MatchSection,
        sorted_regions: List,
    ) -> None:
        """Split a MatchSection containing multiple StructuredRegions into
        per-region child MatchSections.

        Each child receives spans derived from its region's ``RegionPart.span``
        (which carries the exact OCR line ranges from the source JSON).  This
        means the standard BFS path — ``process_fields`` then
        ``process_regions`` — handles each child naturally:

        * ``process_fields`` extracts annotation-based fields scoped to the
          child's OCR lines.
        * ``process_regions`` finds exactly one region via strict containment
          (``region_start >= start_line and region_end <= end_line``), so it
          follows the normal single-region code path with no re-splitting.

        The parent is converted to WRAPPER so the rendering visitor skips it
        while still traversing its children.
        """
        self.logger.debug(
            f"Multi-region MatchSection detected: {len(sorted_regions)} regions "
            f"in section '{section.label}'. Splitting into per-region children."
        )

        for region in sorted_regions:
            child_spans = [part.span for part in region.parts]
            if not child_spans:
                self.logger.warning(
                    f"Region {region.region_id} has no parts; skipping."
                )
                continue

            child = MatchSection()
            child.type = MatchSectionType.CONTENT
            child.owner_layer = section.owner_layer
            child.label = f"{section.label}::region-{region.region_id}"
            child.span = child_spans
            child.parent = section
            child.tags["_split_child"] = True

            section.add_section(child)
            self.logger.debug(
                f"Created child MatchSection '{child.label}' with spans "
                f"{[(s.page, s.y, s.h) for s in child_spans]}"
            )

        # Convert parent to WRAPPER so renderer skips it
        section.type = MatchSectionType.WRAPPER
        section.matched_non_repeating_fields = None
        section.matched_field_rows = None
        self.logger.debug(
            f"Converted section '{section.label}' to WRAPPER with "
            f"{len(section.sections)} per-region children."
        )

    def _process_region_as_kv(
        self,
        context: ExecutionContext,
        regions_cfg: List[Dict],
        match_section: MatchSection,
        structured_section: Section,
        field_mappings: List[FieldMapping],
        all_field_mappings: List[FieldMapping],
        populated_fields: Optional[set[str]] = None,
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

        Args:
            populated_fields: Optional set of field names that have already been populated.
                              When provided, fields in this set will be skipped to prevent
                              duplicate extraction across multiple regions in the same MatchSection.
                              This set is updated in-place as fields are extracted.

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
        # Use the passed-in populated_fields set if provided, otherwise create a new one
        # When shared across multiple sections, this prevents duplicate field extraction
        if populated_fields is None:
            populated_fields = set()
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

                # Two-pass match: prefer exact match over substring/contains
                # This prevents "MEMBER_ID" from matching "MEMBER_ID_NAME"
                exact_generator = (
                    (value_text, it, sel)
                    for sel in selectors
                    for key_text, value_text, it in kv_triplets
                    if key_text and self._ci(sel) == self._ci(key_text)
                )
                first_match = next(exact_generator, None)

                if not first_match:
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

                self.logger.debug(
                    f"Extracting KV field `{field_name}` = '{value_text}' via selector '{matched_selector}' (key='{matched_item.key}')"
                )

                # Resolve field definition
                # Priority: 1. Layer mappings, 2. Global config field definitions
                field_def = template_field_mappings.get(field_name, {}) or {}
                if not field_def:
                    # Look up from global config fields (non_repeating then repeating)
                    if context.conf is not None:
                        config_fields = context.conf.get("fields", {})
                        if field_name in config_fields.get("non_repeating", {}):
                            field_def = OmegaConf.to_container(
                                config_fields.non_repeating[field_name], resolve=True
                            )
                        elif field_name in config_fields.get("repeating", {}):
                            field_def = OmegaConf.to_container(
                                config_fields.repeating[field_name], resolve=True
                            )
                    if not field_def:
                        self.logger.warning(
                            f"No field definition found for '{field_name}', using defaults"
                        )
                        field_def = {}

                field_def = dict(field_def)  # shallow copy
                field_def["name"] = field_name
                # Use ALPHA as default, not MONEY - most text fields are alphanumeric
                field_def.setdefault("type", "ALPHA")

                transformed_value = transform_field_value(field_def, value_text)
                # this is a dummy line_with_meta; we don't have line-level metadata for KV values
                faux_line_with_meta = LineWithMeta(
                    line=value_text,
                    metadata=None,
                    annotations=[],
                )

                created_fields = self.create_fields(
                    field_def, value_text, transformed_value, faux_line_with_meta
                )
                extracted_fields.extend(created_fields)

                # Mark this field as populated to prevent duplicate extraction
                # from other regions in the same MatchSection
                populated_fields.add(field_name)

        # ---- Qualified selector fallback for KV fields ----
        # Fields with qualified selectors (SOURCE:section_path format) that were
        # not populated by regular KV block matching can be resolved from the
        # source record JSON attached as a tag on the section.
        qs_fields = {
            fn: selectors
            for fn, (selectors, _) in kv_specs.items()
            if fn not in populated_fields
            and any(self._parse_qualified_selector(s) for s in selectors)
        }
        if qs_fields:
            qs_source_json = structured_section.tags.get("source_record_json")
            qs_source_record = json.loads(qs_source_json) if qs_source_json else None
            if qs_source_record:
                qs_count = 0
                for field_name, selectors in qs_fields.items():
                    for sel in selectors:
                        parsed = self._parse_qualified_selector(sel)
                        if not parsed:
                            continue
                        source_name, section_path = parsed
                        resolved = self._resolve_qualified_selector(
                            field_name, section_path, qs_source_record
                        )
                        if resolved is None:
                            continue

                        # Resolve field definition
                        field_def = template_field_mappings.get(field_name, {}) or {}
                        if not field_def:
                            if context.conf is not None:
                                config_fields = context.conf.get("fields", {})
                                if field_name in config_fields.get("non_repeating", {}):
                                    field_def = OmegaConf.to_container(
                                        config_fields.non_repeating[field_name],
                                        resolve=True,
                                    )
                                elif field_name in config_fields.get("repeating", {}):
                                    field_def = OmegaConf.to_container(
                                        config_fields.repeating[field_name],
                                        resolve=True,
                                    )
                            if not field_def:
                                field_def = {}

                        field_def = dict(field_def)
                        field_def["name"] = field_name
                        field_def.setdefault("type", "ALPHA")

                        transformed_value = transform_field_value(field_def, resolved)
                        faux_line = LineWithMeta(
                            line=resolved, metadata=None, annotations=[]
                        )
                        created = self.create_fields(
                            field_def, resolved, transformed_value, faux_line
                        )
                        extracted_fields.extend(created)
                        populated_fields.add(field_name)
                        qs_count += len(created)

                        self.logger.debug(
                            f"Extracting KV field `{field_name}` = '{resolved}' "
                            f"via qualified selector '{sel}'"
                        )
                        break  # Found a match, stop trying selectors for this field

                if qs_count:
                    self.logger.info(
                        f"qualified_selector(kv): resolved {qs_count} field(s) "
                        f"for section '{structured_section.title}'"
                    )

        # ---- value_lookup for KV fields ----
        # Fields with a value_lookup config that were not populated by
        # selector matching can be resolved from the source record (the
        # raw claim-extract JSON attached as a tag on the section).
        vl_fields = {
            fn: cfg
            for fn, cfg in fields_cfg.items()
            if isinstance(cfg, dict)
            and "value_lookup" in cfg
            and fn not in populated_fields
        }
        if vl_fields:
            source_record_json = structured_section.tags.get("source_record_json")
            source_record = (
                json.loads(source_record_json) if source_record_json else None
            )
            vl_count = 0
            for field_name, field_cfg in vl_fields.items():
                vl_cfg = field_cfg["value_lookup"]
                source_path = vl_cfg.get("source", "")
                strategy = vl_cfg.get("strategy", "fill_empty")

                # Resolve value
                source_value: Optional[str] = None
                if source_path.startswith("$"):
                    source_value = self._resolve_jsonpath(source_path, source_record)
                else:
                    # Simple dot-path: look up from already-extracted KV fields
                    source_field_name = (
                        source_path.rsplit(".", 1)[-1]
                        if "." in source_path
                        else source_path
                    )
                    for ef in extracted_fields:
                        if ef.field_name == source_field_name and ef.value:
                            source_value = ef.value
                            break

                if not source_value:
                    self.logger.debug(
                        f"value_lookup(kv): source '{source_path}' not found or empty for field '{field_name}'"
                    )
                    continue

                # Resolve field definition for the target field
                field_def: Dict[str, Any] = {}
                if context.conf is not None:
                    config_fields = context.conf.get("fields", {})
                    if field_name in config_fields.get("non_repeating", {}):
                        field_def = OmegaConf.to_container(
                            config_fields.non_repeating[field_name], resolve=True
                        )
                    elif field_name in config_fields.get("repeating", {}):
                        field_def = OmegaConf.to_container(
                            config_fields.repeating[field_name], resolve=True
                        )
                field_def = dict(field_def) if field_def else {}
                field_def["name"] = field_name
                field_def.setdefault("type", "MONEY")

                transformed_value = transform_field_value(field_def, source_value)
                faux_line = LineWithMeta(
                    line=source_value, metadata=None, annotations=[]
                )
                created = self.create_fields(
                    field_def, source_value, transformed_value, faux_line
                )
                extracted_fields.extend(created)
                populated_fields.add(field_name)
                vl_count += len(created)

                self.logger.debug(
                    f"value_lookup(kv): resolved '{field_name}' = '{source_value}' from '{source_path}'"
                )

            if vl_count:
                self.logger.debug(
                    f"value_lookup(kv): filled {vl_count} field(s) for section '{structured_section.title}'"
                )

        # Attach kv fields to the matched section.
        # TODO: we will change this to a dictionary of field types
        if match_section.matched_non_repeating_fields is None:
            match_section.matched_non_repeating_fields = []
        match_section.matched_non_repeating_fields.extend(extracted_fields)

    def _process_region_as_table(
        self,
        document: UnstructuredDocument,
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
        grouping_config = table_config.get('body', {}).get('grouping', {})
        row_types_config = grouping_config.get('row_types', None)
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

        # Now process each TableBlock
        self.logger.debug(
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
            self.logger.debug(f"Processing table block for page: {page_id}")

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

                # Pass 1: prefer exact match over substring/contains
                for selector in selectors:
                    for col_index, header_text in enumerate(header_texts):
                        if col_index in claimed_columns or not header_text:
                            continue
                        if self._ci(selector) == self._ci(header_text):
                            self.logger.debug(
                                f"Matched header '{selector}' for field '{field_name}' at column {col_index} "
                                f"(header='{header_text}', exact)"
                            )
                            processed_column = col_index
                            claimed_columns.add(col_index)
                            matched = True
                            break
                    if matched:
                        break

                # Pass 2: fall back to substring/regex match
                if not matched:
                    for selector in selectors:
                        for col_index, header_text in enumerate(header_texts):
                            if col_index in claimed_columns or not header_text:
                                continue
                            if self._selector_matches_text(
                                selector, header_text, use_regex_flag
                            ):
                                self.logger.debug(
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

            # Add virtual columns for value_lookup fields that did not match
            # any header.  These carry no physical cell — they are populated
            # later by _apply_value_lookup from the source record.
            # Also include columns that are targets of another column's
            # value_lookup.derived_fields (they need a Field stub in each
            # row so the derived-fields distribution can fill them).
            derived_targets: set = set()
            for _fn, _cc in columns_cfg.items():
                if isinstance(_cc, dict):
                    vl = _cc.get("value_lookup")
                    if isinstance(vl, dict):
                        df = vl.get("derived_fields")
                        if isinstance(df, dict):
                            derived_targets.update(df.values())

            for field_name, header_cfg in field_to_header_map.items():
                if field_name in columns_to_process:
                    continue
                col_cfg = columns_cfg.get(field_name, {})
                if isinstance(col_cfg, dict) and (
                    "value_lookup" in col_cfg or field_name in derived_targets
                ):
                    columns_to_process[field_name] = {
                        "cell_index": -1,  # virtual: no physical cell
                        "header_config": header_cfg,
                    }
                    self.logger.info(
                        f"Added virtual column for value_lookup field '{field_name}'"
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

            self.logger.debug(f"Columns to process mapping: {columns_to_process}")

            # Detect primary column and ROW_TYPE column indices for multiline row support
            primary_col_index = -1
            type_col_index = -1

            for field_name, col_def in columns_to_process.items():
                if col_def["header_config"].get("primary", False):
                    primary_col_index = col_def["cell_index"]

            # ROW_TYPE is a classification column from the LLM output — it's typically
            # NOT defined in the config columns, so look for it directly in header_texts
            if row_types_config:
                type_column_name = row_types_config.get("type_column", "ROW_TYPE")
                for col_index, header_text in enumerate(header_texts):
                    if (
                        header_text
                        and header_text.strip().upper() == type_column_name.upper()
                    ):
                        type_col_index = col_index
                        break

            if row_types_config:
                self.logger.debug(
                    f"Row types config detected: primary_col_index={primary_col_index}, "
                    f"type_col_index={type_col_index}, config={row_types_config}"
                )

            # DEBUG: Log region info and body_rows count
            region_id = structured_section.tags.get("source_region_id", "unknown")
            self.logger.debug(
                f"TABLE DEBUG: Processing region '{region_id}' with {len(body_rows)} body_rows for page {page_id}"
            )

            # Extract rows using the resolved column indices
            matched_field_rows: List[MatchFieldRow] = self._build_matched_field_rows(
                document=document,
                body_rows=body_rows,
                columns_to_process=columns_to_process,
                page_id=page_id,
                template_fields_repeating=template_fields_repeating,
                primary_col_index=primary_col_index,
                type_col_index=type_col_index,
                row_types_config=row_types_config,
            )

            if not match_section_to_populate.matched_field_rows:
                match_section_to_populate.matched_field_rows = matched_field_rows
                self.logger.debug(
                    f"ROWS DEBUG: Assigned {len(matched_field_rows)} rows to MatchSection '{match_section_to_populate.label}' "
                    f"(id={id(match_section_to_populate)}). Total rows now: {len(match_section_to_populate.matched_field_rows)}"
                )
            else:  # MatchSection has collected rows from a previous region in scope
                self.logger.debug(
                    f"ROWS DEBUG: Extending MatchSection '{match_section_to_populate.label}' (id={id(match_section_to_populate)}) "
                    f"with {len(matched_field_rows)} rows. Had: {len(match_section_to_populate.matched_field_rows)}"
                )
                match_section_to_populate.matched_field_rows.extend(matched_field_rows)
                self.logger.debug(
                    f"ROWS DEBUG: Now has: {len(match_section_to_populate.matched_field_rows)} rows"
                )

            # Apply value_lookup (claim-level → service-line fallback)
            dist_columns = {
                fn: cfg
                for fn, cfg in columns_cfg.items()
                if isinstance(cfg, dict) and "value_lookup" in cfg
            }
            if dist_columns:
                # Recover the raw source record from the section tag so
                # that JSONPath-style value_lookup sources (``$.…``) can
                # resolve against the full claim record.
                source_record_json = structured_section.tags.get("source_record_json")
                source_record = (
                    json.loads(source_record_json) if source_record_json else None
                )
                self._apply_value_lookup(
                    match_section_to_populate,
                    columns_cfg,
                    matched_field_rows,
                    source_record=source_record,
                    document=document,
                )

    def _apply_value_lookup(
        self,
        match_section: MatchSection,
        columns_cfg: Dict[str, Dict],
        matched_field_rows: List[MatchFieldRow],
        source_record: Optional[Dict[str, Any]] = None,
        document: Optional[UnstructuredDocument] = None,
    ) -> None:
        """Look up values from non-repeating fields, the source record,
        or cross-region resolvers to fill table row fields.

        For each column config with a ``value_lookup`` entry:

        - **Simple dot-path** (e.g. ``claim_information.CLAIM_REMARK_CODE``):
          resolved from ``matched_non_repeating_fields``.
        - **JSONPath** (starts with ``$``, e.g.
          ``$.adjustments[?(@.reason_code == "OA-23")].amount``):
          resolved from the raw *source_record* dict using ``jsonpath_ng``.
        - **Region source** (``region:<role_hint>``): delegates to a pluggable
          resolver function that maps region data to per-row values.
        """
        non_repeating = match_section.matched_non_repeating_fields or []

        if not matched_field_rows:
            return

        # Build lookup: field_name -> value from non-repeating fields
        nr_lookup: Dict[str, str] = {}
        for f in non_repeating:
            if f.field_name and f.value:
                nr_lookup[f.field_name] = f.value

        distributed_count = 0
        for field_name, col_cfg in columns_cfg.items():
            dist_cfg = (
                col_cfg.get("value_lookup") if isinstance(col_cfg, dict) else None
            )
            if not dist_cfg:
                continue

            source_path = dist_cfg.get("source", "")
            strategy = dist_cfg.get("strategy", "fill_empty")

            # ----- Region-based resolver (cross-region lookup) -----
            if source_path.startswith("region:"):
                role_hint = source_path.split(":", 1)[1]
                resolver_path = dist_cfg.get("resolver", "")
                args = dist_cfg.get("args", [])
                derived_fields = dist_cfg.get("derived_fields", None)

                if document and resolver_path:
                    regions = document.regions_by_role(role_hint)
                    if regions:
                        resolver_fn = self._import_resolver(resolver_path)
                        lookup_map = resolver_fn(regions, dist_cfg, matched_field_rows)

                        for row_idx, row in enumerate(matched_field_rows):
                            row_key = self._get_row_match_key(
                                row, args, source_record, row_idx
                            )
                            if row_key and row_key in lookup_map:
                                resolved_value = lookup_map[row_key]

                                if derived_fields and isinstance(resolved_value, dict):
                                    # Derived-fields mode: resolver returned a dict
                                    # — distribute each key to its target column.
                                    for (
                                        derived_key,
                                        target_col,
                                    ) in derived_fields.items():
                                        derived_val = resolved_value.get(derived_key)
                                        if derived_val is None:
                                            continue
                                        for field in row.fields:
                                            if field.field_name != target_col:
                                                continue
                                            if strategy == "fill_empty" and field.value:
                                                continue
                                            field.value = derived_val
                                            if not field.value_original:
                                                field.value_original = derived_val
                                            distributed_count += 1
                                else:
                                    # Legacy mode: resolver returned a plain str.
                                    for field in row.fields:
                                        if field.field_name != field_name:
                                            continue
                                        if strategy == "fill_empty" and field.value:
                                            continue
                                        field.value = resolved_value
                                        if not field.value_original:
                                            field.value_original = resolved_value
                                        distributed_count += 1
                    else:
                        self.logger.debug(
                            f"value_lookup: no regions found for role '{role_hint}'"
                        )
                continue  # Skip normal dot-path/JSONPath resolution

            # ----- Resolve the source value -----
            if source_path.startswith("$"):
                # JSONPath mode: resolve from the raw claim record
                source_value = self._resolve_jsonpath(source_path, source_record)
            else:
                # Simple dot-path mode (existing behaviour)
                # "claim_information.CLAIM_REMARK_CODE" → field_name
                source_field_name = (
                    source_path.rsplit(".", 1)[-1]
                    if "." in source_path
                    else source_path
                )
                source_value = nr_lookup.get(source_field_name)

            if not source_value:
                self.logger.debug(
                    f"value_lookup: source '{source_path}' not found or empty"
                )
                continue

            for row in matched_field_rows:
                for field in row.fields:
                    if field.field_name != field_name:
                        continue
                    if strategy == "fill_empty" and field.value:
                        continue  # Target already has a value, skip
                    field.value = source_value
                    if not field.value_original:
                        field.value_original = source_value
                    distributed_count += 1

        if distributed_count:
            self.logger.info(f"value_lookup: filled {distributed_count} field(s)")

    @staticmethod
    def _resolve_jsonpath(path: str, data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Resolve a JSONPath expression against a record dict.

        Uses ``jsonpath_ng.ext`` which supports filter expressions::

            $.adjustments[?(@.reason_code == "OA-23" && @.type == "CLAIMS_ADJUSTMENT")].amount

        Returns the first matching scalar as a string, or ``None``.
        """
        if not path or not data:
            return None

        try:
            from jsonpath_ng.ext import parse as jsonpath_parse

            # jsonpath_ng.ext uses single ``&`` for logical AND in filters;
            # normalize the common ``&&`` shorthand from YAML configs.
            normalized = path.replace("&&", "&")
            expr = jsonpath_parse(normalized)
            matches = [match.value for match in expr.find(data)]
            if matches:
                return str(matches[0])
        except Exception as e:
            # Import available at module level for type-checking, but
            # guard runtime so a bad expression doesn't crash the pipeline.
            import logging

            logging.warning(f"value_lookup JSONPath error for '{path}': {e}")

        return None

    @staticmethod
    def _import_resolver(dotted_path: str):
        """Dynamically import a resolver function from a dotted module path.

        Args:
            dotted_path: Fully qualified path like
                ``service.extract.core.resolvers.line_resolver``

        Returns:
            The callable resolver function.
        """
        module_path, func_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def _get_row_match_key(
        self,
        row: MatchFieldRow,
        args: List[str],
        source_record: Optional[Dict[str, Any]],
        row_idx: int,
    ) -> Optional[str]:
        """Get the match key for a row using a selector list.

        Each selector in *args* is tried in order:

        1. As a **field name** — resolved from the row's matched fields.
        2. As a **column name** — resolved from ``source_record``
           (e.g. ``service_lines.rows[row_idx]``).
        3. As a **literal value** — used as-is.

        The first selector that produces a non-empty value wins.
        If *args* is empty, falls back to 1-based row index.
        """
        for selector in args:
            # 1. Try from matched row fields
            for f in row.fields:
                if f.field_name == selector and f.value:
                    return f.value.strip()

            # 2. Try from source_record
            if source_record:
                sl = source_record.get("service_lines", {})
                columns = sl.get("columns", [])
                rows = sl.get("rows", [])
                if selector in columns and row_idx < len(rows):
                    col_idx = columns.index(selector)
                    val = (
                        rows[row_idx][col_idx] if col_idx < len(rows[row_idx]) else None
                    )
                    if val:
                        return str(val).strip()

            # 3. Literal — use the selector string directly
            return selector

        # Fallback: 1-based index
        return str(row_idx + 1)

    @staticmethod
    def _parse_qualified_selector(selector: str) -> Optional[tuple[str, str]]:
        """Parse a qualified annotation selector in ``SOURCE:section_path`` format.

        Qualified selectors reference data from a specific annotator source's
        section (e.g. ``CLAIM-EXTRACT:claim_information``) rather than matching
        against KV block item keys.

        Returns ``(source_name, section_path)`` if qualified, ``None`` otherwise.
        """
        if ":" not in selector:
            return None
        # Skip regex-hinted selectors (re:pattern)
        if selector.startswith("re:"):
            return None
        parts = selector.split(":", 1)
        source_name, section_path = parts[0].strip(), parts[1].strip()
        if source_name and section_path:
            return source_name, section_path
        return None

    def _resolve_qualified_selector(
        self,
        field_name: str,
        section_path: str,
        source_record: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Resolve a field value from a source record using a qualified selector.

        Navigates to ``source_record[section_path][field_name]`` and extracts
        the value.  The field data may be a plain string or a dict with a
        ``"value"`` key (the standard claim-extract format).
        """
        if not source_record or not section_path:
            return None

        section_data = source_record.get(section_path)
        if not section_data or not isinstance(section_data, dict):
            return None

        field_data = section_data.get(field_name)
        if field_data is None:
            return None

        if isinstance(field_data, dict):
            value = field_data.get("value")
            return str(value) if value is not None else None
        return str(field_data)

    def _build_matched_field_rows(
        self,
        document: UnstructuredDocument,
        body_rows: List[TableRow],
        columns_to_process: Dict[str, Dict[str, Any]],
        page_id: int,
        template_fields_repeating: Dict[str, Any],
        primary_col_index: int = -1,
        type_col_index: int = -1,
        row_types_config: Optional[Dict[str, Any]] = None,
    ) -> List[MatchFieldRow]:
        """
        Build MatchFieldRow list by extracting values from body rows using resolved column indices.

        When row_types_config is provided, enables dual detection:
        - PRIMARY_COLUMN: if the primary column is empty, the row is a child row
        - ROW_TYPE column: classifies which type of child row (e.g., ADJUSTMENT), each with
          its own active_columns that limit which fields are extracted

        Parameters:
            body_rows: list of TableRow objects with role BODY
            columns_to_process: mapping of field name -> { cell_index: int, header_config: dict }
            page_id: page identifier to propagate to line metadata
            template_fields_repeating: field configuration template for repeating fields
            primary_col_index: index of the primary column (-1 if not set)
            type_col_index: index of the ROW_TYPE column (-1 if not set)
            row_types_config: row types configuration dict from grouping config

        Returns:
            List[MatchFieldRow]
        """
        matched_field_rows: List[MatchFieldRow] = []
        if not body_rows or not columns_to_process:
            return matched_field_rows

        # Pre-compute per row-type config (active_columns, action, merge_strategies, column_mapping)
        type_defs: Dict[str, Dict[str, Any]] = {}
        type_active_columns: Dict[str, set] = {}
        type_column_name: Optional[str] = None
        if row_types_config:
            type_column_name = row_types_config.get("type_column", "ROW_TYPE")
            for type_name, type_def in row_types_config.get("types", {}).items():
                upper_name = type_name.upper()
                type_defs[upper_name] = type_def
                active_cols = type_def.get("active_columns", [])
                if active_cols:
                    type_active_columns[upper_name] = set(active_cols)

        # Stable processing order
        ordered_fields = [
            k
            for k, _ in sorted(
                columns_to_process.items(),
                key=lambda item: item[1]["cell_index"],
            )
        ]

        current_parent: Optional[MatchFieldRow] = None

        for row in body_rows:
            cells = row.cells

            # Step 1: Check primary column — is this a child row?
            is_child_row = False
            if primary_col_index >= 0 and row_types_config:
                if primary_col_index < len(cells):
                    primary_cell = cells[primary_col_index]
                    primary_value = ""
                    if primary_cell.lines and len(primary_cell.lines) > 0:
                        primary_value = (primary_cell.lines[0].line or "").strip()
                    else:
                        primary_value = (
                            str(primary_cell) if primary_cell else ""
                        ).strip()
                    is_child_row = primary_value == ""

            # Step 2: If child row, read ROW_TYPE to resolve type config and action
            active_columns: Optional[set] = (
                None  # None = all columns (default/main row)
            )
            child_type_value = ""
            child_type_def: Dict[str, Any] = {}
            if is_child_row and type_col_index >= 0:
                if type_col_index < len(cells):
                    type_cell = cells[type_col_index]
                    if type_cell.lines and len(type_cell.lines) > 0:
                        child_type_value = (
                            (type_cell.lines[0].line or "").strip().upper()
                        )
                    else:
                        child_type_value = (
                            (str(type_cell) if type_cell else "").strip().upper()
                        )

                    child_type_def = type_defs.get(child_type_value, {})

                    # Check action early — discard before extracting any fields
                    action = child_type_def.get("action", "merge")
                    if action == "discard":
                        self.logger.info(
                            f"Discarding child row with ROW_TYPE='{child_type_value}' (action=discard)"
                        )
                        continue

                    if child_type_value in type_active_columns:
                        active_columns = type_active_columns[child_type_value]
                        self.logger.info(
                            f"Child row detected: ROW_TYPE='{child_type_value}', "
                            f"active_columns={active_columns}, action={action}"
                        )
                    else:
                        self.logger.info(
                            f"Child row detected: ROW_TYPE='{child_type_value}', "
                            f"no active_columns filter, action={action}"
                        )

            extracted_cells = []
            self.logger.debug("row : *******************")

            for field_name in ordered_fields:
                # Skip ROW_TYPE column from output — it's a classification signal, not a data field
                if type_column_name and field_name == type_column_name:
                    continue

                # Skip non-active columns for typed child rows
                if active_columns is not None and field_name not in active_columns:
                    continue

                column_def = columns_to_process[field_name]
                column_index = int(column_def["cell_index"])
                header_config = column_def["header_config"]

                # Virtual column (cell_index == -1): value_lookup field with
                # no physical table cell.  Create an empty Field placeholder
                # that _apply_value_lookup will populate later.
                if column_index == -1:
                    field_def = dict(
                        template_fields_repeating.get(field_name, {}) or {}
                    )
                    field_def["name"] = field_name
                    stub_field = Field(
                        field_name=field_name,
                        field_type=field_def.get("type", "MONEY"),
                        is_required=False,
                        value="",
                        value_original="",
                        page=page_id,
                        confidence=1,
                        scrubbed=True,
                    )
                    extracted_cells.append(stub_field)
                    continue

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

                transformed_value: TransformReturnType = transform_field_value(
                    field_def, cell_value, document
                )
                self.logger.debug(f"transformed_value XXX : {transformed_value}")

                fields = self.create_fields(
                    field_def, cell_value, transformed_value, root_line
                )
                self.logger.debug(f"transformed_value fields : {len(fields)}  {fields}")
                extracted_cells.extend(fields)

            if not is_child_row:
                # Parent row — create new MatchFieldRow and track it
                matched_field_row = MatchFieldRow(fields=extracted_cells)
                matched_field_rows.append(matched_field_row)
                if row_types_config:
                    current_parent = matched_field_row
            else:
                # Child row — action/type already resolved in Step 2 (discard handled via continue above)
                action = child_type_def.get("action", "merge")

                if action == "standalone":
                    self.logger.info(
                        f"Emitting child row ROW_TYPE='{child_type_value}' as standalone (action=standalone)"
                    )
                    matched_field_rows.append(MatchFieldRow(fields=extracted_cells))
                else:
                    # action == "merge" (default)
                    if current_parent is not None:
                        type_merge_strategies = child_type_def.get(
                            "merge_strategies", {}
                        )
                        column_mapping = child_type_def.get("column_mapping", {})
                        default_merge = (
                            row_types_config.get("default_merge", "append")
                            if row_types_config
                            else "append"
                        )

                        self._merge_child_fields_into_parent(
                            current_parent,
                            extracted_cells,
                            merge_strategies=type_merge_strategies,
                            column_mapping=column_mapping,
                            default_merge=default_merge,
                        )
                    else:
                        # Orphaned child row (no parent yet) — emit as standalone
                        self.logger.warning(
                            f"Child row ROW_TYPE='{child_type_value}' without parent; emitting as standalone"
                        )
                        matched_field_rows.append(MatchFieldRow(fields=extracted_cells))

        return matched_field_rows

    def _merge_child_fields_into_parent(
        self,
        parent_row: MatchFieldRow,
        child_fields: List[Field],
        merge_strategies: Optional[Dict[str, str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        default_merge: str = "append",
    ) -> None:
        """Merge active-column fields from a child row into the parent row.

        Args:
            parent_row: The parent SERVICE_LINE MatchFieldRow to merge into.
            child_fields: Fields extracted from the child (ADJUSTMENT) row.
            merge_strategies: Per-column merge strategy overrides
                (child_column_name -> "append"|"replace").
            column_mapping: Remap child field names to different parent field names
                (child_column_name -> target_field_name). When a mapping exists, the
                child value is merged into (or creates) the target field instead of
                the same-named field. Useful when child row values should appear as
                a new column in the parent (e.g., REMARK_CODE -> ADJUSTMENT_REMARK_CODE).
            default_merge: Default merge strategy when no per-column override exists.
        """
        if merge_strategies is None:
            merge_strategies = {}
        if column_mapping is None:
            column_mapping = {}

        parent_field_map: Dict[str, Field] = {
            f.field_name: f for f in parent_row.fields
        }

        for child_field in child_fields:
            child_name = child_field.field_name
            # Apply column_mapping: remap child field to a different target name
            target_name = column_mapping.get(child_name, child_name)
            strategy = merge_strategies.get(child_name, default_merge)
            parent_field = parent_field_map.get(target_name)

            child_val = (child_field.value or "").strip()
            if not child_val:
                continue

            if parent_field is not None:
                if strategy == "replace":
                    parent_field.value = child_val
                    child_orig = (child_field.value_original or "").strip()
                    if child_orig:
                        parent_field.value_original = child_orig
                else:  # "append" (default)
                    parent_val = (parent_field.value or "").strip()
                    if parent_val:
                        parent_field.value = f"{parent_val}, {child_val}"
                    else:
                        parent_field.value = child_val

                    child_orig = (child_field.value_original or "").strip()
                    if child_orig:
                        parent_orig = (parent_field.value_original or "").strip()
                        if parent_orig:
                            parent_field.value_original = f"{parent_orig}, {child_orig}"
                        else:
                            parent_field.value_original = child_orig
            else:
                # Parent doesn't have this field — add it (with remapped name)
                if target_name != child_name:
                    child_field.field_name = target_name
                parent_row.fields.append(child_field)
                parent_field_map[target_name] = child_field

        self.logger.info(
            f"Merged {len(child_fields)} child field(s) into parent row "
            f"(strategy: {default_merge}, mapping: {column_mapping})"
        )

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
                start_line = span.start_line_id
                end_line = span.end_line_id

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
                                    self.logger.debug(
                                        f"Matched header '{selector}' for field '{field_name}'"
                                    )
                                    processed_column = cell_index
                                    break

                    if processed_column != -1:
                        columns_to_process[field_name] = {
                            "cell_index": processed_column,
                            "header_config": header_config,
                        }

                self.logger.debug(f"Columns to process mapping: {columns_to_process}")
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
                            self.logger.debug(
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
                            self.logger.debug(
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
                                    self.logger.debug(
                                        f"Footer row detected using flexible match with pattern: {fallback_pattern}."
                                    )
                                    has_footer = True
                                    footer_row = row
                                    break
                        if has_footer:
                            break

                # If a footer is still not found but always_present is True
                if not has_footer and footer_config.get("always_present", False):
                    self.logger.debug(
                        "Footer row detected (default fallback: always present)."
                    )
                    has_footer = True
                    footer_row = rows[-1]  # Assume last row as footer if unspecified

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
                    extracted_footer_fields = []
                    for field_name, footer_def in field_to_footer_map.items():
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
        self.logger.debug(f"Processing layer: {layer.layer_name}")

        # Filter for fields that are defined at the LAYER scope.
        field_mappings_filtered = [
            fm for fm in layer.fields if fm.scope == FieldScope.LAYER
        ]

        if not field_mappings_filtered:
            self.logger.debug("No layer-level fields to process.")

        for span in spans:
            self.logger.debug(f"Processing span: {span}")
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
        self.logger.debug(f"Extracted match fields for section '{section.label}':")
        for field in extracted_fields:
            self.logger.debug(f"  -  {field}")

    def create_fields(
        self,
        field_def: dict[str, Any],
        value: str,
        transformed_value: TransformReturnType,
        line: LineWithMeta,
    ) -> List[Field]:

        page_id = line.metadata.page_id
        field_name = field_def.get("name")
        derived_fields = (
            field_def.get("derived_fields", None)
            if field_def.get("derived_fields", None)
            else None
        )

        fields: List[Field] = []

        def _build_group(mapping: Optional[Dict[str, Any]]) -> None:
            # If we have derived fields, we create a composite parent field and link children via reference_uuid
            composite_field = derived_fields is not None
            reference_uuid = uuid.uuid4() if composite_field else None

            # For composite parents, keep original "value" as the parent value (as before).
            # Otherwise, use transformed_value (or mapping if provided, though mapping is only for derived children).
            parent_src_value = value if composite_field else transformed_value

            parent_field = Field(
                field_name=field_name,
                field_type=field_def.get("type"),
                is_required=False,
                value=stringify(parent_src_value),
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
                scrubbed=True,
                uuid=reference_uuid,
                reference_uuid=None,
                layer_name="main-layer",
            )
            fields.append(parent_field)

            if not composite_field:
                return

            # Create derived child fields linked to the parent
            for derived_key, derived_value_name in derived_fields.items():
                map_value = None
                if isinstance(mapping, dict):
                    map_value = mapping.get(derived_key, None)

                map_values = map_value if isinstance(map_value, list) else [map_value]

                for map_value in map_values:
                    # derived fields can be None if the parsing did not find a value for it
                    # we are skipping those fields
                    if map_value is None:
                        self.logger.debug(f"Derived key '{derived_key}' has no value")
                        continue

                    child_field = Field(
                        field_name=derived_value_name,
                        field_type=None,
                        is_required=False,
                        value=stringify(map_value),
                        value_original=None,
                        composite_field=False,
                        x=0,
                        y=0,
                        width=0,
                        height=0,
                        date_format=None,
                        name_format=None,
                        column_name=derived_value_name,
                        page=page_id,
                        xdpi=300,
                        ydpi=300,
                        confidence=1,
                        scrubbed=True,
                        uuid=None,
                        reference_uuid=reference_uuid,
                        layer_name="main-layer",
                    )
                    fields.append(child_field)

        # Handle TransformReturnType which can be Dict[str, str|None] or List[Dict[str, str|None]]
        if derived_fields:
            if isinstance(transformed_value, list):
                # Build a composite parent + children group for each mapping
                for mapping in transformed_value:
                    _build_group(mapping if isinstance(mapping, dict) else None)
            elif isinstance(transformed_value, dict):
                _build_group(transformed_value)
            else:
                # Fallback: still create a parent + children with None values
                _build_group(None)
        else:
            # No derived fields: create a single non-composite field using transformed_value
            _build_group(None)

        return fields
