"""Visitor that creates MatchSections from extracted claim records.

For layers configured with ``match_section_source.strategy: record_backed``,
this visitor replaces the cutpoint-based MatchSection creation.  Each extracted
record (from ``agent-output/<data_source>/``) becomes one ``CONTENT``
MatchSection, with provenance stored in ``tags``.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.engine.record_backed_match_section_utils import (
    load_extracted_records,
)
from marie.extract.models.definition import Layer
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection, MatchSectionType, SubzeroResult
from marie.extract.models.span import Span

logger = logging.getLogger(__name__)


class RecordBackedMatchSectionBuilderVisitor(BaseProcessingVisitor):
    """Build MatchSections directly from extracted JSON records.

    Activated only for layers where
    ``layer.match_section_source_strategy == "record_backed"``.
    All other layers are left untouched for the cutpoint-based path.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)

    def visit(self, context: ExecutionContext, parent: SubzeroResult) -> None:
        template = context.get_template()
        if not template or not template.layers:
            return

        for layer in template.layers:
            if layer.match_section_source_strategy != "record_backed":
                continue
            self._build_sections_for_layer(context, layer, parent)

    def _build_sections_for_layer(
        self,
        context: ExecutionContext,
        layer: Layer,
        parent: SubzeroResult,
    ) -> None:
        """Load records and create one MatchSection per record."""
        match_section_source = layer.match_section_source or {}
        data_source = match_section_source.get("data_source")

        # Fallback: derive data_source from region_parser config
        if not data_source and layer.regions_config_raw is not None:
            region_parser_cfg = layer.regions_config_raw[0]
            data_source = region_parser_cfg.get("data_source")

        if not data_source:
            raise ValueError(
                f"Layer '{layer.layer_name}' has strategy 'record_backed' "
                f"but no data_source configured in match_section_source or "
                f"region_parser."
            )

        envelope_key = match_section_source.get("envelope_key")
        records_required = match_section_source.get("records_required", False)

        records = load_extracted_records(
            output_dir=str(context.output_dir),
            data_source=data_source,
            envelope_key=envelope_key,
        )

        if not records and records_required:
            raise ValueError(
                f"Layer '{layer.layer_name}': strategy 'record_backed' "
                f"but no records found in data_source '{data_source}'. "
                f"Will NOT fall back to selectors — check data pipeline."
            )

        logger.info(
            f"Building {len(records)} record-backed MatchSections "
            f"for layer '{layer.layer_name}' from '{data_source}'"
        )

        for record in records:
            section = self._create_section_from_record(layer, record, data_source)
            parent.add_section(section)

    def _create_section_from_record(
        self,
        layer: Layer,
        record: Dict[str, Any],
        data_source: str,
    ) -> MatchSection:
        """Create a single CONTENT MatchSection from one extracted record."""
        claim_uid = record.get("claim_uid", "unknown")
        source = record.get("source", {})

        # Build span from primary source
        spans = self._build_spans(record, source)

        section = MatchSection()
        section.type = MatchSectionType.CONTENT
        section.owner_layer = layer
        section.label = f"{layer.layer_name}::{claim_uid}"
        section.span = spans
        section.row_extraction_strategy = layer.row_extraction_strategy

        # Store provenance in tags
        section.tags["match_section_source_strategy"] = "record_backed"
        section.tags["record_uid"] = claim_uid
        section.tags["data_source"] = data_source
        section.tags["source_record_json"] = json.dumps(record)

        logger.debug(
            f"Created MatchSection '{section.label}' with {len(spans)} span(s)"
        )
        return section

    @staticmethod
    def _build_spans(record: Dict[str, Any], source: Dict[str, Any]) -> List[Span]:
        """Build span list from a record's source metadata.

        If the record has ``_aggregated_sources`` (multi-page claim),
        build spans from all sources.  Otherwise, use the single
        ``source`` entry.
        """
        spans: List[Span] = []

        aggregated = record.get("_aggregated_sources")
        if aggregated and isinstance(aggregated, list):
            for source_index, agg_source in enumerate(aggregated):
                span = _span_from_source(agg_source)
                if span:
                    spans.append(span)
                else:
                    logger.warning(
                        "Record '%s' had unusable source metadata in "
                        "_aggregated_sources[%d]; skipping span",
                        record.get("claim_uid", "?"),
                        source_index,
                    )
        else:
            span = _span_from_source(source)
            if span:
                spans.append(span)

        # Fallback: ensure at least one span so downstream processing works
        if not spans:
            page = source.get("page_index", 0)
            spans.append(Span(page=page, y=0, h=1))
            logger.warning(
                f"Record '{record.get('claim_uid', '?')}' had no usable "
                f"source spans; created fallback span on page {page}"
            )

        return spans


def _span_from_source(source: Dict[str, Any]) -> Optional[Span]:
    """Convert a source dict with ``page_index`` and ``ocr_line_range``
    into a ``Span``."""
    if not isinstance(source, dict) or not source:
        return None

    page = source.get("page_index", 0)
    ocr_range = source.get("ocr_line_range")
    if ocr_range and isinstance(ocr_range, (list, tuple)) and len(ocr_range) >= 2:
        try:
            y = int(ocr_range[0])
            h = max(int(ocr_range[1]) - y, 1)
        except (TypeError, ValueError):
            return None
        return Span(page=page, y=y, h=h)

    return None
