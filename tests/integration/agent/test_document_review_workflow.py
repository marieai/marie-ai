"""Integration tests for the document review workflow example."""

import pytest

from examples.agents.document_review_workflow import (
    SAMPLE_REQUESTS,
    run_document_review_workflow,
)


@pytest.mark.asyncio
async def test_safe_request_is_approved():
    decision = await run_document_review_workflow(
        SAMPLE_REQUESTS["safe"].model_copy(deep=True)
    )

    assert decision.decision == "approve"
    assert decision.selected_specialists == ["extraction", "policy", "risk", "history"]
    assert decision.missing_evidence == []
    assert len(decision.debate) == 2


@pytest.mark.asyncio
async def test_risky_request_is_escalated():
    decision = await run_document_review_workflow(
        SAMPLE_REQUESTS["risky"].model_copy(deep=True)
    )

    assert decision.decision == "escalate"
    assert "history" in decision.selected_specialists
    assert any(citation.startswith("policy:") for citation in decision.citations)
    assert any(citation.startswith("risk:") for citation in decision.citations)
    assert decision.missing_evidence
