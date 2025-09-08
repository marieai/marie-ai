from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from marie.extract.parser.base_region_parser import BaseRegionParser
from marie.extract.structures.structured_region import (
    KeyValue,
    KVList,
    Section,
    SectionRole,
    StructuredRegion,
    ValueType,
)


class JsonRegionParser(BaseRegionParser):
    """
    Generalized JSON -> StructuredRegion parser.

    Responsibilities:
    - Parse JSON into sections based on top-level keys.
    - Parse specific sections into KV lists or tables based on a configurable mapping.
    - Build a StructuredRegion with a PageSpan footprint.
    - Build a single- or multi-page TableSeries via a configurable split policy.

    Notes:
    - Uses the same semantics as MarkdownRegionParser but for JSON input.
    - JSON keys become section titles.
    - Handles both tabular data (with columns/rows) and key-value data.
    """

    def build_single_page_region(
        self,
        json_data: Union[str, Dict[str, Any]],
        *,
        region_id: str,
        page: int,
        page_y: int,
        page_h: int,
    ) -> StructuredRegion:
        """
        Parse JSON into a single-page StructuredRegion. All table rows live on the given page.
        """
        return super().build_single_page_region(
            data=json_data,
            region_id=region_id,
            page=page,
            page_y=page_y,
            page_h=page_h,
        )

    def build_multi_page_region(
        self,
        json_data: Union[str, Dict[str, Any]],
        *,
        region_id: str,
        # Page 1 footprint
        p1_page: int,
        p1_y: int,
        p1_h: int,
        # Page 2 footprint
        p2_page: int,
        p2_y: int,
        p2_h: int,
    ) -> StructuredRegion:
        """
        Parse JSON into a two-page StructuredRegion (generic multi-page demo).
        """
        return super().build_multi_page_region(
            data=json_data,
            region_id=region_id,
            p1_page=p1_page,
            p1_y=p1_y,
            p1_h=p1_h,
            p2_page=p2_page,
            p2_y=p2_y,
            p2_h=p2_h,
        )

    def _parse_and_validate_sections(
        self, json_data: Union[str, Dict[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """
        Common entry to parse sections and enforce validation rules:
        - Parse JSON string or accept dict directly
        - Extract top-level keys as section titles
        - No duplicate titles (shouldn't happen with JSON keys but validate anyway)
        """
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        elif isinstance(json_data, dict):
            data = json_data
        else:
            raise ValueError("json_data must be a JSON string or dictionary")

        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object/dictionary")

        sections_seq = [(key.upper(), value) for key, value in data.items()]
        self._validate_sections(sections_seq)
        return sections_seq

    def _parse_table_content(self, data: Any) -> Tuple[List[str], List[List[str]]]:
        """
        Parse JSON data into table format (headers, rows).
        Expects either:
        1. {"columns": [...], "rows": [[...], [...]]}
        2. [{"col1": "val1", "col2": "val2"}, ...]  (array of objects)
        3. {"key1": "val1", "key2": "val2"}  (single object, treated as one row)
        """
        if not data:
            return [], []

        # Format 1: Explicit columns and rows structure
        if isinstance(data, dict) and "columns" in data and "rows" in data:
            columns = data["columns"]
            rows = data["rows"]
            if not isinstance(columns, list) or not isinstance(rows, list):
                return [], []

            # Convert all row values to strings
            string_rows = []
            for row in rows:
                if isinstance(row, list):
                    string_row = [str(cell) if cell is not None else "" for cell in row]
                    string_rows.append(string_row)

            return [str(col) for col in columns], string_rows

        # Format 2: Array of objects
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Extract headers from first object
            headers = list(data[0].keys())
            rows = []
            for obj in data:
                if isinstance(obj, dict):
                    row = [str(obj.get(header, "")) for header in headers]
                    rows.append(row)
            return headers, rows

        # Format 3: Single object (treated as single row)
        if isinstance(data, dict) and "columns" not in data:
            headers = list(data.keys())
            row = [str(data[header]) for header in headers]
            return headers, [row]

        return [], []

    def _build_kv_section(
        self, title_uc: str, role: SectionRole, data: Any
    ) -> Optional["Section"]:
        """
        Build a KVList Section if data is a dict and allowed by kv_section_titles.
        """
        key = self._norm_title_key(title_uc)
        has_cfg = getattr(self, "_has_sections_cfg", False)

        # Enforce filtering: if strict config is present, only allow explicitly configured KV sections.
        if has_cfg:
            if key not in self.kv_section_titles:
                return None
        else:
            # Backward-compat: if no strict config, honor optional allow-list if provided
            if self.kv_section_titles and key not in self.kv_section_titles:
                return None

        if not isinstance(data, dict):
            return None

        # Convert dict to KeyValue objects
        kv_items = []
        for k, v in data.items():
            # Convert value to string representation
            if isinstance(v, (str, int, float, bool)):
                value_str = str(v)
            elif v is None:
                value_str = "None"
            else:
                value_str = json.dumps(v, ensure_ascii=False)

            kv_items.append(
                KeyValue(
                    key=k,
                    value=value_str,
                    value_type=ValueType.UNKNOWN,
                )
            )

        if not kv_items:
            return None

        return Section(
            title=self._as_title(title_uc),
            role=role,
            blocks=[KVList(type="kvlist", items=kv_items)],
        )
