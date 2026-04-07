"""Integration tests for the document review workflow example."""

import json

import pytest

from examples.agents.document_review_workflow import (
    SAMPLE_REQUESTS,
    DebateTurn,
    DecisionEnvelope,
    DocumentReviewWorkflow,
    SpecialistFinding,
    run_document_review_workflow,
)
from marie.agent.message import Message


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


class FakeLLM:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.calls = 0

    async def achat(self, messages, functions=None, extra_generate_cfg=None):
        del messages, functions, extra_generate_cfg
        payload = self.payloads[self.calls]
        self.calls += 1
        return Message.assistant(json.dumps(payload))


@pytest.mark.asyncio
async def test_live_mode_returns_structured_decision_with_fake_llm():
    request = SAMPLE_REQUESTS["safe"].model_copy(deep=True)
    fake_llm = FakeLLM(
        [
            SpecialistFinding(
                specialist="extraction",
                summary="Fields resolved.",
                evidence=["Amount found", "Signature present"],
                citations=["document:amount"],
                risk_score=0,
            ).model_dump(),
            SpecialistFinding(
                specialist="policy",
                summary="Policy check passed.",
                evidence=["Under threshold"],
                citations=["policy:auto-approval-limit"],
                risk_score=0,
            ).model_dump(),
            SpecialistFinding(
                specialist="risk",
                summary="No manual-review phrases found.",
                evidence=["No high-risk indicators"],
                citations=["risk:phrase-scan"],
                risk_score=0,
            ).model_dump(),
            SpecialistFinding(
                specialist="history",
                summary="No concerning history found.",
                evidence=["No prior flags"],
                citations=["history:prior-flags"],
                risk_score=0,
            ).model_dump(),
            DebateTurn(
                agent="approve_agent",
                position="approve",
                argument="The packet is clear for approval.",
                evidence_refs=["policy:auto-approval-limit"],
                confidence=0.81,
            ).model_dump(),
            DebateTurn(
                agent="challenge_agent",
                position="escalate",
                argument="Escalation is not strongly justified.",
                evidence_refs=[],
                confidence=0.31,
            ).model_dump(),
            DecisionEnvelope(
                document_id=request.document_id,
                decision="approve",
                confidence=0.83,
                summary="Approve the packet.",
                next_action="Approve automatically and retain the decision.",
                missing_evidence=[],
                citations=[
                    "document:amount",
                    "policy:auto-approval-limit",
                    "risk:phrase-scan",
                    "history:prior-flags",
                ],
                selected_specialists=["extraction", "policy", "risk", "history"],
                specialist_findings=[],
                debate=[],
            ).model_dump(),
        ]
    )

    workflow = DocumentReviewWorkflow(mode="live", llm=fake_llm)
    decision = await workflow.run(request)

    assert decision.decision == "approve"
    assert decision.selected_specialists == ["extraction", "policy", "risk", "history"]
    assert len(decision.specialist_findings) == 4
    assert len(decision.debate) == 2
    assert fake_llm.calls == 7
