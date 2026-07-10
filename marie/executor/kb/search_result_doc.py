"""Doc schema for vector/hybrid search results.

Search result metadata (similarity, source_id, node_type, ...) must be
declared pydantic fields rather than attached via `doc.metadata = {...}`
on a stock `TextDoc`: `TextDoc` has no `metadata` field, so that
assignment fails under docarray's `validate_assignment`, and even if
bypassed, undeclared attributes do not survive `DocList` protobuf
serialization across the worker -> gateway boundary.
"""

from __future__ import annotations

from typing import Optional

from docarray.documents import TextDoc


class SearchResultDoc(TextDoc):
    """TextDoc extended with search result ranking/provenance fields."""

    similarity: float = 0.0
    text_score: Optional[float] = None
    rrf_score: Optional[float] = None
    source_id: str = ""
    node_type: str = ""
    index_name: str = ""
    ref_doc_id: Optional[str] = None
