import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from marie.extract.annotators.context_provider import ContextProvider, ProcessingUnit
from marie.extract.registry import register_context_provider
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.logging_core.predefined import default_logger as logger

if TYPE_CHECKING:
    from marie_kernel.context import RunContext

    from marie.extract.structures.unstructured_document import UnstructuredDocument


class TableContextProvider(ContextProvider):
    """
    Base provider for table-based context injection.

    Retrieves table extraction results from upstream tasks and injects
    them into prompt templates.

    Subclasses should override `_should_include_table()` to filter tables.
    """

    SOURCE_TASK: str = "tables"

    def __init__(
        self,
        run_context: Optional["RunContext"],
        annotator_name: str,
    ):
        super().__init__(run_context, annotator_name)
        self._tables_by_page: Dict[int, List[Dict[str, Any]]] = {}
        self._load_tables()

    def _should_include_table(self, table: Dict[str, Any]) -> bool:
        """
        Determine if a table should be included in the context.

        Override in subclasses for custom filtering logic.
        Default: include all tables.

        Args:
            table: Table extraction dict with keys like 'table_classification', etc.

        Returns:
            True if table should be included, False otherwise.
        """
        return True

    def _load_tables(self) -> None:
        """Load and cache table data from upstream task results."""
        if not self.run_context:
            logger.debug(
                f"No run_context available for {self.__class__.__name__}, "
                "no tables will be loaded"
            )
            return

        tables_data = self.run_context.get_annotation(self.SOURCE_TASK)
        if not tables_data:
            logger.debug(
                f"No annotation data found for source_task '{self.SOURCE_TASK}'"
            )
            return

        # Extract tables from the annotation data structure
        # Expected structure: { "pages": { page_num: { "data": { "extractions": [...] } } } }
        pages_data = tables_data.get("pages", {})
        if not pages_data:
            logger.debug("No pages data in table annotation")
            return

        for page_num_str, page_data in pages_data.items():
            try:
                page_num = int(page_num_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid page number: {page_num_str}")
                continue

            extractions = page_data.get("data", {}).get("extractions", [])
            if not extractions:
                continue

            # Filter using subclass-defined logic
            filtered = [t for t in extractions if self._should_include_table(t)]

            # Sort tables by their starting line number for deterministic order
            # This ensures consistent _t0, _t1, etc. assignment across runs
            def get_table_start_line(table: Dict[str, Any]) -> int:
                """Get the first header line number for sorting."""
                header_rows = table.get("header_rows", [])
                if header_rows and isinstance(header_rows[0], dict):
                    return header_rows[0].get("line_number", float("inf"))
                return float("inf")

            filtered.sort(key=get_table_start_line)

            if filtered:
                self._tables_by_page[page_num] = filtered
                logger.debug(
                    f"Loaded {len(filtered)} tables for page {page_num} "
                    f"(sorted by line: {[get_table_start_line(t) for t in filtered]})"
                )

        logger.info(
            f"{self.__class__.__name__}: Loaded tables for {len(self._tables_by_page)} pages"
        )

    def get_eligible_pages(self, document: "UnstructuredDocument") -> Set[int]:
        """
        Return pages that have table data available.

        Args:
            document: The document being processed.

        Returns:
            Set of 1-indexed page numbers with table data.
        """
        return set(self._tables_by_page.keys())

    def get_processing_units(
        self, document: "UnstructuredDocument"
    ) -> List[ProcessingUnit]:
        """
        Return one ProcessingUnit per table (not per page).

        This enables per-table LLM calls instead of bundling all tables
        on a page into a single call.

        Args:
            document: The document being processed.

        Returns:
            List of ProcessingUnit, one for each table found.
        """
        units = []
        for page_num in sorted(self._tables_by_page.keys()):
            tables = self._tables_by_page[page_num]
            for idx, table in enumerate(tables):
                units.append(
                    ProcessingUnit(
                        page_number=page_num,
                        index=idx,
                        data=table,
                    )
                )
        return units

    def get_tables_json(self, page_number: int) -> str:
        """Get JSON-formatted table data for a page."""
        tables = self._tables_by_page.get(page_number, [])
        return json.dumps(tables, indent=2) if tables else ""


@register_context_provider(
    name="table_claims",
    target_annotators=["claim-extract", "claim-validation"],
)
class TableClaimContextProvider(TableContextProvider):
    """
    Provider for CLAIM table context injection.

    Combines both table data and pre-annotated claim context for claim-extract.

    Injects variables:
    - TABLE_CONTEXT_CLAIMS: JSON of table data
    - TABLE_INDEX: Index of table within page
    - TABLE_NAME: Name of table if available
    - INFERRED_HEADERS_HINT: Column mapping hint for headerless continuation tables
    - CLAIM_CONTEXT: Pre-annotated claims closest to this table
    - CLAIM_COUNT: Number of claims found
    - HAS_CLAIMS: "true" or "false"
    - CLAIM_SOURCE_PAGE: Page where claims were found

    Auto-activates for annotators: claim-extract, claim-validation
    """

    def _should_include_table(self, table: Dict[str, Any]) -> bool:
        return table.get("table_classification") == "CLAIM"

    def _build_inferred_headers_hint(self, columns: List[str]) -> str:
        """
        Build a hint string for headerless tables.

        When a table has no visible headers (continuation tables), this provides
        explicit column mapping instructions to the LLM.

        Args:
            columns: List of column names inferred from previous page.

        Returns:
            Formatted hint string for LLM prompt injection.
        """
        hint = """
## HEADERLESS TABLE - Column Order

**This table has NO visible headers.** The columns are inferred from a previous page.
Use this EXACT column order for extraction (left to right):

"""
        for i, col in enumerate(columns, 1):
            hint += f"{i}. {col}\n"

        hint += """
Map data values to columns by POSITION, not by header matching.
"""
        return hint

    def _extract_claims_from_document(
        self, document: "UnstructuredDocument"
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract CLAIM annotations from document lines.

        Returns dict mapping page_number (1-indexed) to list of claim dicts.
        """
        claims_by_page: Dict[int, List[Dict[str, Any]]] = {}

        for page_id in range(document.page_count):
            page_claims = []
            lines = document.lines_for_page(page_id)

            for line in lines:
                if not line.annotations:
                    continue

                for annotation in line.annotations:
                    if isinstance(annotation, TypedAnnotation):
                        if annotation.annotation_type == "CLAIM":
                            page_claims.append(
                                {
                                    "line_number": line.metadata.line_id,
                                    "label": annotation.name,
                                    "value": annotation.value,
                                }
                            )

            if page_claims:
                claims_by_page[page_id + 1] = page_claims  # 1-indexed

        return claims_by_page

    def _get_claims_for_table(
        self,
        claims_by_page: Dict[int, List[Dict[str, Any]]],
        page_number: int,
        table_header_line: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Get claims relevant to this table based on header line position.

        Strategy:
        1. If table has header line, get claims on same page BEFORE that line
        2. If no claims found (or no header line), fall back to previous pages

        Args:
            claims_by_page: Dict mapping page numbers to claim lists.
            page_number: 1-indexed page number of the table.
            table_header_line: Line number of the table header (optional).

        Returns:
            Tuple of (claims_list, source_page_number).
            source_page_number is the page where claims were found,
            or 0 if no claims found anywhere.
        """
        # Get claims on current page
        page_claims = claims_by_page.get(page_number, [])

        if page_claims and table_header_line is not None:
            # Filter to claims BEFORE the table header
            relevant = [c for c in page_claims if c["line_number"] < table_header_line]
            if relevant:
                return relevant, page_number

        # If claims exist on page (no header filter or all after header), use them
        if page_claims:
            return page_claims, page_number

        # Fallback: search previous pages (closest first)
        for prev_page in range(page_number - 1, 0, -1):
            if prev_page in claims_by_page:
                logger.debug(
                    f"No claims on page {page_number}, using claims from page {prev_page}"
                )
                return claims_by_page[prev_page], prev_page

        # No claims found anywhere
        return [], 0

    def get_variables(
        self,
        document: "UnstructuredDocument",
        page_number: int,
        unit: Optional[ProcessingUnit] = None,
    ) -> Dict[str, str]:
        """Get table AND claim context variables."""
        variables: Dict[str, str] = {}
        table_header_line: Optional[int] = None

        # Table context
        if unit and unit.data is not None:
            # Per-table mode: inject single table
            variables["TABLE_CONTEXT_CLAIMS"] = json.dumps(unit.data, indent=2)
            variables["TABLE_INDEX"] = str(unit.index)
            variables["TABLE_NAME"] = unit.data.get("name", "")

            # Get table header line for claim filtering
            header_rows = unit.data.get("header_rows", [])
            if header_rows:
                first_header = header_rows[0]
                if isinstance(first_header, dict):
                    table_header_line = first_header.get("line_number")

            # Check if headers are missing (continuation table)
            header_present = unit.data.get("header_present", True)
            columns = unit.data.get("columns", [])

            inferred_headers_hint = ""
            if not header_present or not header_rows:
                if columns:
                    inferred_headers_hint = self._build_inferred_headers_hint(columns)

            variables["INFERRED_HEADERS_HINT"] = inferred_headers_hint
        else:
            # Legacy mode: inject all tables for page
            variables["TABLE_CONTEXT_CLAIMS"] = self.get_tables_json(page_number)
            variables["INFERRED_HEADERS_HINT"] = ""

        # # Claim context
        # claims_by_page = self._extract_claims_from_document(document)
        # claims, source_page = self._get_claims_for_table(
        #     claims_by_page, page_number, table_header_line
        # )
        #
        # variables["CLAIM_CONTEXT"] = json.dumps(claims, indent=2) if claims else ""
        # variables["CLAIM_COUNT"] = str(len(claims))
        # variables["HAS_CLAIMS"] = "true" if claims else "false"
        # variables["CLAIM_SOURCE_PAGE"] = str(source_page) if source_page > 0 else ""

        return variables


@register_context_provider(
    name="table_remark_codes",
    target_annotators=["remarks"],
)
class TableRemarkCodesContextProvider(TableContextProvider):
    """
    Provider for CODES/REMARKS table context injection.

    Injects multiple variables:
    - TABLE_CONTEXT_CODES: JSON of single table (per-table mode) or all tables (legacy)
    - TABLE_COUNT: Number of tables found
    - HAS_TABLES: "true" or "false"
    - TABLE_INDEX: Index of the table within the page (per-table mode only)
    - TABLE_NAME: Name of the table if available (per-table mode only)

    Auto-activates for annotators: remarks
    """

    def _should_include_table(self, table: Dict[str, Any]) -> bool:
        classification = table.get("table_classification")
        return classification in ("CODES", "REMARKS") if classification else False

    def get_variables(
        self,
        document: "UnstructuredDocument",
        page_number: int,
        unit: Optional[ProcessingUnit] = None,
    ) -> Dict[str, str]:
        if unit and unit.data is not None:
            # Per-table mode: inject single table
            return {
                "TABLE_CONTEXT_CODES": json.dumps(unit.data, indent=2),
                "TABLE_COUNT": "1",
                "HAS_TABLES": "true",
                "TABLE_INDEX": str(unit.index),
                "TABLE_NAME": unit.data.get("name", ""),
            }
        else:
            # Legacy mode: inject all tables for page
            tables = self._tables_by_page.get(page_number, [])
            return {
                "TABLE_CONTEXT_CODES": self.get_tables_json(page_number),
                "TABLE_COUNT": str(len(tables)),
                "HAS_TABLES": "true" if tables else "false",
            }
