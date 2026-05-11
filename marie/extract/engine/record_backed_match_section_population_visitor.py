"""Visitor that populates record-backed MatchSections from extracted data.

For MatchSections tagged with ``match_section_source_strategy == "record_backed"``,
this visitor reads the claim record stored in ``tags["source_record_json"]`` and
populates KV fields and table rows directly — bypassing the region-overlap and
selector-matching pipeline.
"""

import importlib
import json
import logging
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.engine.record_backed_match_section_utils import (
    normalize_record_value,
)
from marie.extract.engine.transform import TransformReturnType, transform_field_value
from marie.extract.models.definition import FieldMapping
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    Field,
    MatchFieldRow,
    MatchSection,
    MatchSectionType,
    SubzeroResult,
)
from marie.extract.structures.line_metadata import LineMetadata
from marie.extract.structures.line_with_meta import LineWithMeta

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Collapse whitespace in a value for field storage."""
    import re

    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()


class RecordBackedMatchSectionPopulationVisitor(BaseProcessingVisitor):
    """Populate record-backed MatchSections from their stored claim records.

    Walks the MatchSection tree (BFS) and, for each ``CONTENT`` section
    tagged with ``match_section_source_strategy == "record_backed"``,
    extracts KV fields and table rows directly from the JSON record
    attached in ``tags["source_record_json"]``.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)

    def visit(self, context: ExecutionContext, parent: SubzeroResult) -> None:
        logger.debug("Processing RecordBackedMatchSectionPopulationVisitor")
        queue = deque([parent])
        while queue:
            current = queue.popleft()
            if current is None:
                continue
            if (
                current.type == MatchSectionType.CONTENT
                and current.tags.get("match_section_source_strategy") == "record_backed"
            ):
                self._process_section(context, parent, current)
            queue.extend(current.sections)
        logger.debug("Finished RecordBackedMatchSectionPopulationVisitor")

    def _process_section(
        self,
        context: ExecutionContext,
        parent: MatchSection,
        section: MatchSection,
    ) -> None:
        """Process a single record-backed MatchSection."""
        layer = section.owner_layer
        if layer is None or layer.regions_config_raw is None:
            logger.warning(
                f"Section '{section.label}' has no layer or regions_config_raw; skipping."
            )
            return

        # Unpack layer config (same tuple as used by the legacy visitor)
        region_parser_cfg, regions_cfg, template_fields_repeating = (
            layer.regions_config_raw
        )
        region_parser_cfg = OmegaConf.to_container(region_parser_cfg, resolve=True)
        regions_cfg = OmegaConf.to_container(regions_cfg, resolve=True)
        template_fields_repeating = OmegaConf.to_container(
            template_fields_repeating, resolve=True
        )

        # Load the claim record from the section's tags
        source_record_json = section.tags.get("source_record_json")
        if not source_record_json:
            logger.warning(
                f"Section '{section.label}' has no source_record_json; skipping."
            )
            return

        claim_record = json.loads(source_record_json)
        parser_sections_rules = region_parser_cfg.get("sections", [])

        field_mappings: List[FieldMapping] = layer.non_repeating_field_mappings

        # Track populated KV fields to avoid duplicates
        populated_fields: set = set()

        for section_rule in parser_sections_rules:
            role = section_rule.get("role")
            parse_method = section_rule.get("parse")
            section_title = section_rule.get("title", role or "")

            if not role:
                continue

            if parse_method == "kv":
                self._populate_kv(
                    context=context,
                    regions_cfg=regions_cfg,
                    match_section=section,
                    claim_record=claim_record,
                    role=role,
                    section_title=section_title,
                    field_mappings=field_mappings,
                    populated_fields=populated_fields,
                )
            elif parse_method == "table":
                self._populate_table(
                    context=context,
                    regions_cfg=regions_cfg,
                    match_section=section,
                    claim_record=claim_record,
                    role=role,
                    section_title=section_title,
                    template_fields_repeating=template_fields_repeating,
                )
            else:
                logger.debug(
                    f"Skipping section rule with parse method '{parse_method}' "
                    f"for role '{role}'"
                )

    # ------------------------------------------------------------------
    # KV population
    # ------------------------------------------------------------------

    def _populate_kv(
        self,
        context: ExecutionContext,
        regions_cfg: List[Dict],
        match_section: MatchSection,
        claim_record: Dict[str, Any],
        role: str,
        section_title: str,
        field_mappings: List[FieldMapping],
        populated_fields: set,
    ) -> None:
        """Populate non-repeating (KV) fields from the claim record."""
        # Find region entry by title
        region_entry = _find_region_entry(regions_cfg, section_title, "kv")
        if region_entry is None:
            logger.debug(
                f"No 'kv' region config found for section '{section_title}'; skipping."
            )
            return

        fields_cfg = region_entry.get("fields", {}) or {}
        if not fields_cfg:
            return

        claim_data = claim_record.get(role, {})
        if not claim_data or not isinstance(claim_data, dict):
            logger.debug(f"No data for role '{role}' in claim record; skipping KV.")
            return

        # Build template field mapping lookup
        template_field_mappings: Dict[str, Dict] = {}
        for mapping in field_mappings:
            if mapping.field_def:
                template_field_mappings[mapping.field_def["name"]] = mapping.field_def

        page_id = _get_page_id(claim_record)
        extracted_fields: List[Field] = []

        for field_name, field_info in fields_cfg.items():
            if field_name in populated_fields:
                continue

            # Get selectors — these are the source keys in the JSON
            selectors = _get_selectors(field_info)
            if not selectors:
                continue

            # Match against claim_data keys (case-insensitive)
            matched_value = self._match_kv_value(selectors, claim_data)

            if matched_value is None:
                continue

            logger.debug(
                f"Extracting KV field '{field_name}' = '{matched_value}' "
                f"from record role '{role}'"
            )

            # Resolve field definition
            field_def = _resolve_field_def(field_name, template_field_mappings, context)
            field_def.setdefault("type", "ALPHA")

            transformed_value = transform_field_value(field_def, matched_value)
            faux_line = LineWithMeta(line=matched_value, metadata=LineMetadata(page_id, None, None), annotations=[])
            created_fields = _create_fields(
                field_def, matched_value, transformed_value, faux_line
            )
            extracted_fields.extend(created_fields)
            populated_fields.add(field_name)

        # ---- Qualified selector fallback ----
        self._populate_kv_qualified_selectors(
            context=context,
            fields_cfg=fields_cfg,
            claim_record=claim_record,
            template_field_mappings=template_field_mappings,
            populated_fields=populated_fields,
            extracted_fields=extracted_fields,
        )

        # ---- value_lookup for KV fields ----
        self._populate_kv_value_lookup(
            context=context,
            fields_cfg=fields_cfg,
            claim_record=claim_record,
            extracted_fields=extracted_fields,
            populated_fields=populated_fields,
        )

        # Attach to section
        if match_section.matched_non_repeating_fields is None:
            match_section.matched_non_repeating_fields = []
        match_section.matched_non_repeating_fields.extend(extracted_fields)

    def _match_kv_value(
        self, selectors: List[str], claim_data: Dict[str, Any]
    ) -> Optional[str]:
        """Find first matching key in claim_data (case-insensitive)."""
        claim_keys_lower = {k.casefold(): k for k in claim_data}

        for sel in selectors:
            # Skip qualified selectors — handled separately
            if ":" in sel and not sel.startswith("re:"):
                continue
            matched_key = claim_keys_lower.get(sel.casefold())
            if matched_key is not None:
                return normalize_record_value(claim_data[matched_key])
        return None

    def _populate_kv_qualified_selectors(
        self,
        context: ExecutionContext,
        fields_cfg: Dict,
        claim_record: Dict[str, Any],
        template_field_mappings: Dict,
        populated_fields: set,
        extracted_fields: List[Field],
    ) -> None:
        """Resolve fields with qualified selectors (SOURCE:section_path)."""
        page_id = _get_page_id(claim_record)
        for field_name, field_info in fields_cfg.items():
            if field_name in populated_fields:
                continue

            selectors = _get_selectors(field_info)
            for sel in selectors:
                parsed = _parse_qualified_selector(sel)
                if not parsed:
                    continue
                _source_name, section_path = parsed
                resolved = _resolve_qualified_selector(
                    field_name, section_path, claim_record
                )
                if resolved is None:
                    continue

                field_def = _resolve_field_def(
                    field_name, template_field_mappings, context
                )
                field_def.setdefault("type", "ALPHA")

                transformed_value = transform_field_value(field_def, resolved)
                faux_line = LineWithMeta(line=resolved, metadata=LineMetadata(page_id, None, None), annotations=[])
                created = _create_fields(
                    field_def, resolved, transformed_value, faux_line
                )
                extracted_fields.extend(created)
                populated_fields.add(field_name)
                break

    def _populate_kv_value_lookup(
        self,
        context: ExecutionContext,
        fields_cfg: Dict,
        claim_record: Dict[str, Any],
        extracted_fields: List[Field],
        populated_fields: set,
    ) -> None:
        """Resolve KV fields that use value_lookup configuration."""
        vl_fields = {
            fn: cfg
            for fn, cfg in fields_cfg.items()
            if isinstance(cfg, dict)
            and "value_lookup" in cfg
            and fn not in populated_fields
        }
        if not vl_fields:
            return

        page_id = _get_page_id(claim_record)
        for field_name, field_cfg in vl_fields.items():
            vl_cfg = field_cfg["value_lookup"]
            source_path = vl_cfg.get("source", "")

            source_value: Optional[str] = None
            if source_path.startswith("$"):
                source_value = _resolve_jsonpath(source_path, claim_record)
            else:
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
                continue

            field_def = _resolve_field_def(field_name, {}, context)
            field_def.setdefault("type", "MONEY")

            transformed_value = transform_field_value(field_def, source_value)
            faux_line = LineWithMeta(line=source_value, metadata=LineMetadata(page_id, None, None), annotations=[])
            created = _create_fields(
                field_def, source_value, transformed_value, faux_line
            )
            extracted_fields.extend(created)
            populated_fields.add(field_name)

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(
        self,
        context: ExecutionContext,
        regions_cfg: List[Dict],
        match_section: MatchSection,
        claim_record: Dict[str, Any],
        role: str,
        section_title: str,
        template_fields_repeating: Dict,
    ) -> None:
        """Populate repeating (table) fields from the claim record."""
        region_entry = _find_region_entry(regions_cfg, section_title, "table")
        if region_entry is None:
            logger.debug(
                f"No 'table' region config found for '{section_title}'; skipping."
            )
            return

        table_body_config = region_entry.get("table", {}).get("body", {}) or {}
        columns_cfg = table_body_config.get("columns", {}) or {}
        if not columns_cfg:
            return

        table_data = claim_record.get(role, {})
        if not table_data or not isinstance(table_data, dict):
            logger.debug(f"No table data for role '{role}' in claim record; skipping.")
            return

        json_columns = table_data.get("columns", [])
        json_rows = table_data.get("rows", [])

        if not json_columns or not json_rows:
            logger.debug(f"Empty columns or rows for role '{role}'; skipping.")
            return

        # Build column index map: field_name -> (col_index, col_cfg)
        json_col_lower = {c.casefold(): idx for idx, c in enumerate(json_columns)}
        column_map: Dict[str, Dict[str, Any]] = {}

        # Detect derived_fields targets (need stub Fields)
        derived_targets: set = set()
        for _fn, _cc in columns_cfg.items():
            if isinstance(_cc, dict):
                vl = _cc.get("value_lookup")
                if isinstance(vl, dict):
                    df = vl.get("derived_fields")
                    if isinstance(df, dict):
                        derived_targets.update(df.values())

        for field_name, col_cfg in columns_cfg.items():
            selectors = _get_selectors(col_cfg)
            col_index = -1

            for sel in selectors:
                idx = json_col_lower.get(sel.casefold())
                if idx is not None:
                    col_index = idx
                    break

            if col_index >= 0:
                column_map[field_name] = {
                    "col_index": col_index,
                    "config": col_cfg,
                }
            elif isinstance(col_cfg, dict) and (
                "value_lookup" in col_cfg or field_name in derived_targets
            ):
                # Virtual column — no physical cell, populated by value_lookup
                column_map[field_name] = {
                    "col_index": -1,
                    "config": col_cfg,
                }

        if not column_map:
            logger.warning(f"No columns mapped for table role '{role}'; skipping.")
            return

        # ROW_TYPE
        row_types_config = table_body_config.get("grouping", {}).get("row_types")
        type_defs: Dict[str, Dict[str, Any]] = {}
        type_active_columns: Dict[str, set] = {}
        type_col_index = -1
        primary_col_index = -1

        if row_types_config:
            type_col_name = row_types_config.get("type_column", "ROW_TYPE")
            type_col_index = json_col_lower.get(type_col_name.casefold(), -1)

            for t_name, t_def in row_types_config.get("types", {}).items():
                upper_name = t_name.upper()
                type_defs[upper_name] = t_def
                if "active_columns" in t_def:
                    type_active_columns[upper_name] = set(t_def["active_columns"])

            logger.debug(
                f"Row types config detected: primary_col_index={primary_col_index}, "
                f"type_col_index={type_col_index}, config={row_types_config}"
            )

        # Detect the primary column index
        for fn, cc in columns_cfg.items():
            if isinstance(cc, dict) and cc.get("primary"):
                primary_col_index = column_map.get(fn, {}).get("col_index", -1)
                break

        # Build rows
        matched_field_rows: List[MatchFieldRow] = []
        page_id = _get_page_id(claim_record)

        current_parent: Optional[MatchFieldRow] = None

        for row_data in json_rows:
            # Detect child row
            is_child_row = False
            if primary_col_index >= 0 and row_types_config:
                prim_val = str(row_data[primary_col_index]).strip() if primary_col_index < len(row_data) else ""
                is_child_row = (prim_val == "")

            # Resolve type and action
            child_type_value = ""
            child_type_def: Dict[str, Any] = {}
            active_columns: Optional[set] = None  # None = all columns (default/main row)

            if is_child_row and type_col_index >= 0:
                child_type_value = str(row_data[type_col_index]).strip().upper() if type_col_index < len(row_data) else ""
                child_type_def = type_defs.get(child_type_value, {})

                action = child_type_def.get("action", "merge")
                if action == "discard":
                    logger.debug(f"Discarding child row with ROW_TYPE='{child_type_value}'")
                    continue

                active_columns = type_active_columns.get(child_type_value)
                if active_columns is None:
                    logger.debug(
                        f"Child row detected: ROW_TYPE='{child_type_value}', "
                        f"active_columns={active_columns}, action={action}"
                    )
                else:
                    logger.debug(
                        f"Child row detected: ROW_TYPE='{child_type_value}', "
                        f"no active_columns filter, action={action}"
                    )

            row_fields: List[Field] = []

            # Extract columns
            for field_name, col_def in column_map.items():
                # Skip ROW_TYPE classification column from output
                if row_types_config and field_name == row_types_config.get("type_column", "ROW_TYPE"):
                    continue

                # Skip non-active columns for typed child rows
                if active_columns is not None and field_name not in active_columns:
                    continue

                col_index = col_def["col_index"]
                col_cfg = col_def["config"]

                # Virtual column — stub field for value_lookup
                if col_index == -1:
                    field_def = dict(
                        template_fields_repeating.get(field_name, {}) or {}
                    )
                    field_def["name"] = field_name
                    stub = Field(
                        field_name=field_name,
                        field_type=field_def.get("type", "MONEY"),
                        is_required=False,
                        value="",
                        value_original="",
                        page=page_id,
                        confidence=1,
                        scrubbed=True,
                    )
                    row_fields.append(stub)
                    continue

                # Extract cell value
                if col_index < len(row_data):
                    raw_value = row_data[col_index]
                    cell_value = str(raw_value) if raw_value is not None else ""
                else:
                    cell_value = ""

                field_def = dict(template_fields_repeating.get(field_name, {}) or {})
                field_def["name"] = field_name

                transformed_value = transform_field_value(field_def, cell_value)
                faux_line = LineWithMeta(line=cell_value, metadata=LineMetadata(page_id, None, None), annotations=[])
                created = _create_fields(
                    field_def, cell_value, transformed_value, faux_line
                )
                row_fields.extend(created)

            if not is_child_row:
                # Parent row — create new MatchFieldRow and track it
                matched_row = MatchFieldRow(fields=row_fields)
                matched_field_rows.append(matched_row)
                if row_types_config:
                    current_parent = matched_row
            else:
                # Child row — action/type already resolved in Step 2 (discard handled via continue above)
                action = child_type_def.get("action", "merge")
                if action == "standalone" or current_parent is None:
                    logger.debug(f"Emitting child row ROW_TYPE='{child_type_value}' as standalone (action=standalone)")
                    matched_field_rows.append(MatchFieldRow(fields=row_fields))
                else: # action == "merge" (default)
                    merge_strategies = child_type_def.get("merge_strategies", {})
                    column_mapping = child_type_def.get("column_mapping", {})
                    default_merge = row_types_config.get("default_merge", "append") if row_types_config else "append"

                    self._merge_child_fields_into_parent(
                        current_parent,
                        row_fields,
                        merge_strategies,
                        column_mapping,
                        default_merge
                    )

        # Apply value_lookup for table columns
        dist_columns = {
            fn: cfg
            for fn, cfg in columns_cfg.items()
            if isinstance(cfg, dict) and "value_lookup" in cfg
        }
        if dist_columns:
            self._apply_value_lookup(
                match_section=match_section,
                columns_cfg=columns_cfg,
                matched_field_rows=matched_field_rows,
                source_record=claim_record,
                document=context.document,
            )

        # Attach rows to section
        if not match_section.matched_field_rows:
            match_section.matched_field_rows = matched_field_rows
        else:
            match_section.matched_field_rows.extend(matched_field_rows)

        logger.info(
            f"Populated {len(matched_field_rows)} table rows for "
            f"section '{match_section.label}' role '{role}'"
        )

    # ------------------------------------------------------------------
    # row_types (merge table rows) — reused from legacy visitor logic
    # ------------------------------------------------------------------

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

        logger.info(
            f"Merged {len(child_fields)} child field(s) into parent row "
            f"(strategy: {default_merge}, mapping: {column_mapping})"
        )

    # ------------------------------------------------------------------
    # value_lookup (table rows) — reused from legacy visitor logic
    # ------------------------------------------------------------------

    def _apply_value_lookup(
        self,
        match_section: MatchSection,
        columns_cfg: Dict[str, Dict],
        matched_field_rows: List[MatchFieldRow],
        source_record: Optional[Dict[str, Any]] = None,
        document=None,
    ) -> None:
        """Look up values to fill table row fields.

        Supports:
        - Simple dot-path from non-repeating fields
        - JSONPath from source record
        - Region-based cross-region resolvers
        - Section-based within same region
        """
        non_repeating = match_section.matched_non_repeating_fields or []

        if not matched_field_rows:
            return

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

            # Region-based resolver
            if source_path.startswith("region:"):
                role_hint = source_path.split(":", 1)[1]
                resolver_path = dist_cfg.get("resolver", "")
                args = dist_cfg.get("args", [])
                derived_fields = dist_cfg.get("derived_fields", None)

                if document and resolver_path:
                    regions = document.regions_by_role(role_hint)
                    if regions:
                        resolver_fn = _import_resolver(resolver_path)
                        lookup_map = resolver_fn(regions, dist_cfg, matched_field_rows)

                        distributed_count = distribute_resolved_values(
                            matched_field_rows,
                            lookup_map,
                            args,
                            source_record,
                            derived_fields,
                            strategy,
                            field_name,
                            distributed_count,
                            _get_row_match_key,
                        )
                continue

            if source_path.startswith("section:"):
                role_hint = source_path.split(":", 1)[1]
                resolver_path = dist_cfg.get("resolver", "")
                args = dist_cfg.get("args", [])
                derived_fields = dist_cfg.get("derived_fields", None)

                if source_record and resolver_path:
                    sections = source_record.get(role_hint, [])
                    if sections:
                        resolver_fn = _import_resolver(resolver_path)
                        lookup_map = resolver_fn(sections, dist_cfg, matched_field_rows)

                        distributed_count = distribute_resolved_values(
                            matched_field_rows,
                            lookup_map,
                            args,
                            source_record,
                            derived_fields,
                            strategy,
                            field_name,
                            distributed_count,
                            _get_row_match_key,
                        )
                    continue
            # JSONPath or dot-path resolution
            if source_path.startswith("$"):
                source_value = _resolve_jsonpath(source_path, source_record)
            else:
                source_field_name = (
                    source_path.rsplit(".", 1)[-1]
                    if "." in source_path
                    else source_path
                )
                source_value = nr_lookup.get(source_field_name)

            if not source_value:
                continue

            for row in matched_field_rows:
                for field in row.fields:
                    if field.field_name != field_name:
                        continue
                    if strategy == "fill_empty" and field.value:
                        continue
                    field.value = source_value
                    if not field.value_original:
                        field.value_original = source_value
                    distributed_count += 1

        if distributed_count:
            logger.info(f"value_lookup: filled {distributed_count} field(s)")


# ======================================================================
# Module-level helper functions (shared / duplicated from legacy visitor)
# ======================================================================

def distribute_resolved_values(
        matched_field_rows,
        lookup_map,
        args,
        source_record,
        derived_fields,
        strategy,
        field_name,
        distributed_count,
        _get_row_match_key,
):
    """
    Distributes resolved values to fields in matched_field_rows.
    - If derived_fields is provided and resolved_value is a dict, distributes per derived field.
    - Otherwise, distributes the resolved_value to the field matching field_name.
    Returns the updated distributed_count.
    """
    for row_idx, row in enumerate(matched_field_rows):
        row_key = _get_row_match_key(row, args, source_record, row_idx)
        if not row_key or row_key not in lookup_map:
            continue
        resolved_value = lookup_map[row_key]

        if derived_fields and isinstance(resolved_value, dict):
            for derived_key, target_col in derived_fields.items():
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
            for field in row.fields:
                if field.field_name != field_name:
                    continue
                if strategy == "fill_empty" and field.value:
                    continue
                field.value = resolved_value
                if not field.value_original:
                    field.value_original = resolved_value
                distributed_count += 1
    return distributed_count

def _find_region_entry(
    regions_cfg: List[Dict], section_title: str, expected_type: str
) -> Optional[Dict]:
    """Find a region entry by title and type."""
    title_upper = (section_title or "").strip().upper()
    for entry in regions_cfg:
        if str(entry.get("title", "")).strip().upper() == title_upper:
            if entry.get("type") == expected_type:
                return entry
    return None


def _get_selectors(field_info: Any) -> List[str]:
    """Extract selector list from a field/column config dict."""
    if not isinstance(field_info, dict):
        return []
    if "annotation_selectors" in field_info:
        return [str(s) for s in field_info["annotation_selectors"] if s]
    if "selectors" in field_info:
        return [str(s) for s in field_info["selectors"] if s]
    if "selector" in field_info and field_info["selector"]:
        return [str(field_info["selector"])]
    return []


def _resolve_field_def(
    field_name: str,
    template_field_mappings: Dict[str, Dict],
    context: ExecutionContext,
) -> Dict[str, Any]:
    """Resolve a field definition from layer mappings or global config."""
    field_def = template_field_mappings.get(field_name, {}) or {}
    if not field_def:
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
            field_def = {}

    field_def = dict(field_def)
    field_def["name"] = field_name
    return field_def


def _parse_qualified_selector(selector: str) -> Optional[tuple]:
    """Parse ``SOURCE:section_path`` format. Returns ``(source, path)`` or ``None``."""
    if ":" not in selector:
        return None
    if selector.startswith("re:"):
        return None
    parts = selector.split(":", 1)
    source_name, section_path = parts[0].strip(), parts[1].strip()
    if source_name and section_path:
        return source_name, section_path
    return None


def _resolve_qualified_selector(
    field_name: str,
    section_path: str,
    source_record: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Resolve a field value from ``source_record[section_path][field_name]``."""
    if not source_record or not section_path:
        return None

    section_data = source_record.get(section_path)
    if not section_data or not isinstance(section_data, dict):
        return None

    field_data = section_data.get(field_name)
    return normalize_record_value(field_data)


def _resolve_jsonpath(path: str, data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve a JSONPath expression against a record dict."""
    if not path or not data:
        return None
    try:
        from jsonpath_ng.ext import parse as jsonpath_parse

        normalized = path.replace("&&", "&")
        expr = jsonpath_parse(normalized)
        matches = [match.value for match in expr.find(data)]
        if matches:
            return str(matches[0])
    except Exception as e:
        logger.warning(f"value_lookup JSONPath error for '{path}': {e}")
    return None


def _import_resolver(dotted_path: str):
    """Dynamically import a resolver function."""
    module_path, func_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _get_row_match_key(
    row: MatchFieldRow,
    args: List[str],
    source_record: Optional[Dict[str, Any]],
    row_idx: int,
) -> Optional[str]:
    """Get the match key for a row using a selector list."""
    for selector in args:
        for f in row.fields:
            if f.field_name == selector and f.value:
                return f.value.strip()

        if source_record:
            sl = source_record.get("service_lines", {})
            columns = sl.get("columns", [])
            rows = sl.get("rows", [])
            if selector in columns and row_idx < len(rows):
                col_idx = columns.index(selector)
                val = rows[row_idx][col_idx] if col_idx < len(rows[row_idx]) else None
                if val:
                    return str(val).strip()

        return selector

    return str(row_idx + 1)


def _get_page_id(claim_record: Dict[str, Any]) -> int:
    """Get the page ID from the claim record."""
    # NOTE: page_index may resolve to None. Default to 0.
    return claim_record.get("source", {}).get("page_index", 0) or 0


def _create_fields(
    field_def: Dict[str, Any],
    value: str,
    transformed_value: TransformReturnType,
    line: Optional[LineWithMeta],
) -> List[Field]:
    """Create Field objects, handling derived fields (composite parent + children).

    Mirrors ``MatchSectionExtractionProcessingVisitor.create_fields``.
    """
    page_id = line.metadata.page_id if line and line.metadata else 0
    field_name = field_def.get("name")
    derived_fields = field_def.get("derived_fields") or None
    fields: List[Field] = []

    def _build_group(mapping: Optional[Dict[str, Any]]) -> None:
        composite_field = derived_fields is not None
        reference_uuid = uuid.uuid4() if composite_field else None

        parent_src_value = value if composite_field else transformed_value

        parent_field = Field(
            field_name=field_name,
            field_type=field_def.get("type"),
            is_required=False,
            value=_stringify(parent_src_value),
            value_original=_stringify(value),
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

        for derived_key, derived_value_name in derived_fields.items():
            map_value = None
            if isinstance(mapping, dict):
                map_value = mapping.get(derived_key, None)

            map_values = map_value if isinstance(map_value, list) else [map_value]

            for mv in map_values:
                if mv is None:
                    continue
                child_field = Field(
                    field_name=derived_value_name,
                    field_type=None,
                    is_required=False,
                    value=_stringify(mv),
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

    if derived_fields:
        if isinstance(transformed_value, list):
            for mapping in transformed_value:
                _build_group(mapping if isinstance(mapping, dict) else None)
        elif isinstance(transformed_value, dict):
            _build_group(transformed_value)
        else:
            _build_group(None)
    else:
        _build_group(None)

    return fields
