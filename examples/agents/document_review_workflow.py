"""Production-style multi-agent workflow example for document review.

This example mirrors a common production pattern:

1. Route the request to the right specialists
2. Run independent specialists concurrently
3. Hold a short approval-vs-escalation debate
4. Emit structured JSON for downstream automation

The default mode is deterministic so the example is runnable in CI and local
development without a live model backend. A live mode is also available when
you want the specialists, debate, and judge to use a real provider-backed LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from marie.agent.config import CoordinationConfig
from marie.agent.coordination.fan_out import FanOutCoordinator
from marie.agent.message import Message

AUTO_APPROVAL_LIMIT = 5000.0
HIGH_RISK_VENDORS = {
    "rapid supply llc": "Vendor appears on the watch list for manual payment exceptions.",
    "northwind legacy imports": "Vendor has a history of disputed change requests.",
}
RISK_PHRASES = {
    "manual override": "Manual override language detected.",
    "urgent wire": "Urgent payment request detected.",
    "refund outside cycle": "Refund outside the normal cycle detected.",
    "handwritten": "Handwritten amendment detected.",
}


class WorkflowInput(BaseModel):
    document_id: str
    reviewer_goal: str
    document_text: str
    amount: float = 0.0
    vendor: str = "unknown"
    prior_flags: int = 0
    customer_tier: Literal["standard", "priority", "regulated"] = "standard"


class SpecialistFinding(BaseModel):
    specialist: str
    summary: str
    blockers: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    risk_score: int = 0


class DebateTurn(BaseModel):
    agent: str
    position: Literal["approve", "escalate"]
    argument: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float


class DecisionEnvelope(BaseModel):
    document_id: str
    decision: Literal["approve", "escalate"]
    confidence: float
    summary: str
    next_action: str
    missing_evidence: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    selected_specialists: list[str] = Field(default_factory=list)
    specialist_findings: list[SpecialistFinding] = Field(default_factory=list)
    debate: list[DebateTurn] = Field(default_factory=list)


@dataclass
class SpecialistAgent:
    name: str
    handler: Callable[[WorkflowInput], SpecialistFinding]

    async def arun(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        del messages
        request: WorkflowInput = kwargs["request"]
        finding = self.handler(request)
        return {
            "output": finding.model_dump(),
            "metadata": {"confidence": max(0.25, 1.0 - finding.risk_score * 0.15)},
        }


@dataclass
class LiveSpecialistAgent:
    name: str
    llm: Any

    async def arun(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        del messages
        request: WorkflowInput = kwargs["request"]
        finding = await build_live_specialist_finding(self.llm, self.name, request)
        return {
            "output": finding.model_dump(),
            "metadata": {"confidence": max(0.25, 1.0 - finding.risk_score * 0.15)},
        }


def extract_amount(document_text: str) -> float | None:
    match = re.search(r"\$([0-9][0-9,]*(?:\.\d{2})?)", document_text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def build_extraction_finding(request: WorkflowInput) -> SpecialistFinding:
    amount = request.amount or extract_amount(request.document_text) or 0.0
    lower_text = request.document_text.lower()
    blockers: list[str] = []
    evidence = [
        f"Document amount resolved to ${amount:,.2f}.",
        f"Customer tier is {request.customer_tier}.",
    ]

    has_signature = bool(
        re.search(r"\bsigned\b", lower_text) or "signature on file" in lower_text
    )
    if "unsigned" in lower_text:
        has_signature = False

    if has_signature:
        evidence.append("Signature language is present.")
    else:
        blockers.append("Signed approval marker is missing.")

    return SpecialistFinding(
        specialist="extraction",
        summary="Resolved core document fields for review.",
        blockers=blockers,
        evidence=evidence,
        citations=["document:amount", "document:signature"],
        risk_score=len(blockers),
    )


def build_policy_finding(request: WorkflowInput) -> SpecialistFinding:
    lower_text = request.document_text.lower()
    amount = request.amount or extract_amount(request.document_text) or 0.0
    blockers: list[str] = []
    evidence = [f"Auto-approval limit is ${AUTO_APPROVAL_LIMIT:,.2f}."]
    citations = ["policy:auto-approval-limit"]

    if amount > AUTO_APPROVAL_LIMIT:
        blockers.append("Amount exceeds the auto-approval threshold.")
    else:
        evidence.append("Amount is within the automatic approval threshold.")

    if request.customer_tier == "regulated":
        citations.append("policy:regulated-customer-review")
        if "compliance approved" not in lower_text:
            blockers.append("Regulated customer review requires compliance approval.")
        else:
            evidence.append(
                "Compliance approval is present for the regulated customer."
            )

    return SpecialistFinding(
        specialist="policy",
        summary="Checked the packet against approval controls.",
        blockers=blockers,
        evidence=evidence,
        citations=citations,
        risk_score=len(blockers),
    )


def build_history_finding(request: WorkflowInput) -> SpecialistFinding:
    vendor_key = request.vendor.strip().lower()
    blockers: list[str] = []
    evidence = [f"Prior review flags: {request.prior_flags}."]
    citations = ["history:prior-flags"]

    if request.prior_flags >= 2:
        blockers.append("Requestor has repeated prior review flags.")

    if vendor_key in HIGH_RISK_VENDORS:
        blockers.append(HIGH_RISK_VENDORS[vendor_key])
        citations.append("history:vendor-watch-list")
    else:
        evidence.append(f"Vendor {request.vendor} is not on the watch list.")

    return SpecialistFinding(
        specialist="history",
        summary="Reviewed prior incidents and vendor history.",
        blockers=blockers,
        evidence=evidence,
        citations=citations,
        risk_score=len(blockers) + request.prior_flags,
    )


def build_risk_finding(request: WorkflowInput) -> SpecialistFinding:
    lower_text = request.document_text.lower()
    blockers = [
        detail for phrase, detail in RISK_PHRASES.items() if phrase in lower_text
    ]
    evidence = [f"Matched {len(blockers)} risk phrases in the request."]
    citations = ["risk:phrase-scan"]

    amount = request.amount or extract_amount(request.document_text) or 0.0
    if amount >= 10000:
        blockers.append("Large transaction amount requires manual review.")
        citations.append("risk:large-transaction")

    return SpecialistFinding(
        specialist="risk",
        summary="Scanned for manual-review indicators.",
        blockers=blockers,
        evidence=evidence,
        citations=citations,
        risk_score=len(blockers),
    )


def select_specialists(request: WorkflowInput) -> list[str]:
    specialists = ["extraction", "policy", "risk"]
    if (
        request.prior_flags > 0
        or request.vendor != "unknown"
        or request.customer_tier != "standard"
    ):
        specialists.append("history")
    return specialists


def build_approve_turn(findings: list[SpecialistFinding]) -> DebateTurn:
    clear_findings = [finding for finding in findings if not finding.blockers]
    evidence_refs = [
        finding.citations[0] for finding in clear_findings if finding.citations
    ]
    confidence = min(0.88, 0.56 + len(clear_findings) * 0.08)

    if clear_findings:
        argument = (
            "The packet satisfies the core controls that were checked, and no reviewer "
            "found a hard blocker in those passing lanes."
        )
    else:
        argument = (
            "There is some usable evidence in the packet, but the case for automatic "
            "approval is weak because every specialist raised a concern."
        )

    return DebateTurn(
        agent="approve_agent",
        position="approve",
        argument=argument,
        evidence_refs=evidence_refs[:3],
        confidence=round(confidence, 2),
    )


def build_escalate_turn(findings: list[SpecialistFinding]) -> DebateTurn:
    blockers = [
        f"{finding.specialist}: {blocker}"
        for finding in findings
        for blocker in finding.blockers
    ]
    evidence_refs = [
        finding.citations[0]
        for finding in findings
        if finding.blockers and finding.citations
    ]
    confidence = min(0.94, 0.6 + len(blockers) * 0.06)

    if blockers:
        argument = (
            "The packet should be escalated because multiple specialists found approval "
            "gaps that need a human decision."
        )
    else:
        argument = "No hard blocker was found, so escalation cannot be justified."

    return DebateTurn(
        agent="challenge_agent",
        position="escalate",
        argument=argument,
        evidence_refs=evidence_refs[:4],
        confidence=round(confidence, 2),
    )


def build_missing_evidence(findings: list[SpecialistFinding]) -> list[str]:
    missing: list[str] = []
    blocker_text = " ".join(
        blocker.lower() for finding in findings for blocker in finding.blockers
    )

    if "signed approval marker is missing" in blocker_text:
        missing.append("Signed approval page or signature confirmation")
    if "compliance approval" in blocker_text:
        missing.append("Compliance approval note for the regulated account")
    if "repeated prior review flags" in blocker_text:
        missing.append("Reviewer disposition for the prior flagged requests")
    if "watch list" in blocker_text or "manual review" in blocker_text:
        missing.append("Human reviewer justification for vendor and payment exception")

    return missing


def parse_json_message_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("Expected string or dict content from LLM response.")

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


async def generate_structured_model(
    llm: Any,
    schema: type[BaseModel],
    *,
    system_prompt: str,
    user_prompt: str,
) -> BaseModel:
    response = await llm.achat(
        [
            Message.system(system_prompt),
            Message.user(user_prompt),
        ],
        extra_generate_cfg={"guided_json": schema.model_json_schema()},
    )
    payload = parse_json_message_content(response.content)
    return schema.model_validate(payload)


def render_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    elif isinstance(value, list):
        value = [
            item.model_dump() if isinstance(item, BaseModel) else item for item in value
        ]
    return json.dumps(value, indent=2, sort_keys=True)


def build_live_specialist_prompt(name: str, request: WorkflowInput) -> tuple[str, str]:
    rules = {
        "extraction": (
            "Resolve the packet fields from the request text. Check whether the packet "
            "contains valid signature language. Use blockers for missing required fields "
            "or a missing signature marker."
        ),
        "policy": (
            f"Check the packet against the auto-approval limit of ${AUTO_APPROVAL_LIMIT:,.2f}. "
            "If the customer tier is regulated, require explicit compliance approval in the text."
        ),
        "risk": (
            "Scan the text for manual-review indicators such as manual override, urgent wire, "
            "refund outside cycle, and handwritten changes. Large transaction amounts should "
            "also increase risk."
        ),
        "history": (
            "Review vendor risk and prior review flags. Repeated flags and watch-list vendors "
            "should be treated as blockers."
        ),
    }
    system_prompt = (
        f"You are the {name} specialist in a document review workflow. "
        "Return only JSON that matches the schema."
    )
    user_prompt = (
        f"Specialist: {name}\n"
        f"Task: {rules[name]}\n\n"
        "Workflow input:\n"
        f"{render_json(request)}\n\n"
        "Return a SpecialistFinding. Use the exact specialist name provided."
    )
    return system_prompt, user_prompt


async def build_live_specialist_finding(
    llm: Any, name: str, request: WorkflowInput
) -> SpecialistFinding:
    system_prompt, user_prompt = build_live_specialist_prompt(name, request)
    finding = await generate_structured_model(
        llm,
        SpecialistFinding,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return finding.model_copy(update={"specialist": name})


async def build_live_debate_turn(
    llm: Any,
    position: Literal["approve", "escalate"],
    findings: list[SpecialistFinding],
) -> DebateTurn:
    system_prompt = (
        "You are participating in a short document-review debate. "
        "Return only JSON that matches the schema."
    )
    user_prompt = (
        f"Take the {position} side.\n\n"
        "Specialist findings:\n"
        f"{render_json(findings)}\n\n"
        "Return a DebateTurn. Use approve_agent for approve and challenge_agent for escalate."
    )
    turn = await generate_structured_model(
        llm,
        DebateTurn,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    expected_agent = "approve_agent" if position == "approve" else "challenge_agent"
    return turn.model_copy(update={"agent": expected_agent, "position": position})


async def build_live_decision(
    llm: Any,
    request: WorkflowInput,
    selected_specialists: list[str],
    findings: list[SpecialistFinding],
    debate: list[DebateTurn],
) -> DecisionEnvelope:
    system_prompt = (
        "You are the final judge in a document review workflow. "
        "Return only JSON that matches the schema."
    )
    user_prompt = (
        "Review the workflow input, specialist findings, and debate. "
        "Decide whether the packet should be approved or escalated.\n\n"
        f"Workflow input:\n{render_json(request)}\n\n"
        f"Selected specialists:\n{render_json(selected_specialists)}\n\n"
        f"Specialist findings:\n{render_json(findings)}\n\n"
        f"Debate turns:\n{render_json(debate)}\n\n"
        "Return a DecisionEnvelope."
    )
    decision = await generate_structured_model(
        llm,
        DecisionEnvelope,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return decision.model_copy(
        update={
            "document_id": request.document_id,
            "selected_specialists": selected_specialists,
            "specialist_findings": findings,
            "debate": debate,
        }
    )


def judge(
    request: WorkflowInput,
    selected_specialists: list[str],
    findings: list[SpecialistFinding],
    debate: list[DebateTurn],
) -> DecisionEnvelope:
    blockers = [blocker for finding in findings for blocker in finding.blockers]
    citations = []
    for finding in findings:
        citations.extend(finding.citations)
    citations = list(dict.fromkeys(citations))

    risk_score = sum(finding.risk_score for finding in findings)
    should_escalate = bool(blockers) or risk_score >= 4

    if should_escalate:
        decision = "escalate"
        summary = (
            "Escalate the packet. The specialist review found approval gaps or manual-review "
            "signals that should not be auto-resolved."
        )
        next_action = "Route the packet to a human reviewer with the specialist evidence attached."
        confidence = min(0.97, 0.66 + len(blockers) * 0.05)
    else:
        decision = "approve"
        summary = "Approve the packet. The specialists did not find a blocking policy, history, or risk issue."
        next_action = (
            "Approve automatically and retain the structured decision for audit."
        )
        confidence = 0.74

    return DecisionEnvelope(
        document_id=request.document_id,
        decision=decision,
        confidence=round(confidence, 2),
        summary=summary,
        next_action=next_action,
        missing_evidence=build_missing_evidence(findings) if should_escalate else [],
        citations=citations,
        selected_specialists=selected_specialists,
        specialist_findings=findings,
        debate=debate,
    )


class DocumentReviewWorkflow:
    def __init__(
        self,
        *,
        mode: Literal["deterministic", "live"] = "deterministic",
        backend: str = "openai",
        model: str | None = None,
        llm: Any | None = None,
    ) -> None:
        self.mode = mode
        self.llm = llm
        if self.mode == "live" and self.llm is None:
            from examples.agents.utils import create_llm

            self.llm = create_llm(backend=backend, model=model)

        if self.mode == "live":
            self._agent_builders = {
                "extraction": lambda: LiveSpecialistAgent("extraction", self.llm),
                "policy": lambda: LiveSpecialistAgent("policy", self.llm),
                "risk": lambda: LiveSpecialistAgent("risk", self.llm),
                "history": lambda: LiveSpecialistAgent("history", self.llm),
            }
        else:
            self._agent_builders = {
                "extraction": lambda: SpecialistAgent(
                    "extraction", build_extraction_finding
                ),
                "policy": lambda: SpecialistAgent("policy", build_policy_finding),
                "risk": lambda: SpecialistAgent("risk", build_risk_finding),
                "history": lambda: SpecialistAgent("history", build_history_finding),
            }
        self._coordination_config = CoordinationConfig(
            topology="parallel",
            merge_strategy="aggregate",
            max_concurrent=4,
            timeout=10.0,
        )

    async def run(self, request: WorkflowInput) -> DecisionEnvelope:
        selected_specialists = select_specialists(request)
        coordinator = FanOutCoordinator(self._coordination_config)
        coordinator.add_agents(
            [self._agent_builders[name]() for name in selected_specialists]
        )

        result = await coordinator.run(
            [Message.user(request.reviewer_goal)],
            request=request,
        )
        findings = [
            SpecialistFinding.model_validate(agent_result.output)
            for agent_result in result.results
            if agent_result.output
        ]
        findings.sort(
            key=lambda finding: selected_specialists.index(finding.specialist)
        )

        if self.mode == "live":
            debate = [
                await build_live_debate_turn(self.llm, "approve", findings),
                await build_live_debate_turn(self.llm, "escalate", findings),
            ]
            return await build_live_decision(
                self.llm,
                request,
                selected_specialists,
                findings,
                debate,
            )

        debate = [build_approve_turn(findings), build_escalate_turn(findings)]
        return judge(request, selected_specialists, findings, debate)


SAMPLE_REQUESTS = {
    "safe": WorkflowInput(
        document_id="DOC-1001",
        reviewer_goal="Decide whether the packet should be auto-approved or escalated.",
        document_text=(
            "Invoice INV-2042 for $1,250. Signed by account owner. Delivery confirmation "
            "attached and no exception requested."
        ),
        amount=1250.0,
        vendor="Acme Office Supply",
        prior_flags=0,
        customer_tier="standard",
    ),
    "risky": WorkflowInput(
        document_id="DOC-2009",
        reviewer_goal="Decide whether the packet should be auto-approved or escalated.",
        document_text=(
            "Manual override requested for refund outside cycle. Handwritten note says urgent "
            "wire today. Unsigned adjustment for $12,400."
        ),
        amount=12400.0,
        vendor="Rapid Supply LLC",
        prior_flags=2,
        customer_tier="regulated",
    ),
}


async def run_document_review_workflow(
    request: WorkflowInput,
    *,
    mode: Literal["deterministic", "live"] = "deterministic",
    backend: str = "openai",
    model: str | None = None,
    llm: Any | None = None,
) -> DecisionEnvelope:
    workflow = DocumentReviewWorkflow(
        mode=mode,
        backend=backend,
        model=model,
        llm=llm,
    )
    return await workflow.run(request)


def build_request_from_args(args: argparse.Namespace) -> WorkflowInput:
    if args.sample:
        base = SAMPLE_REQUESTS[args.sample].model_copy(deep=True)
    else:
        base = WorkflowInput(
            document_id=args.document_id,
            reviewer_goal=args.reviewer_goal,
            document_text=args.document_text,
            amount=args.amount,
            vendor=args.vendor,
            prior_flags=args.prior_flags,
            customer_tier=args.customer_tier,
        )

    if args.document_text:
        base.document_text = args.document_text
    if args.amount is not None:
        base.amount = args.amount
    if args.vendor:
        base.vendor = args.vendor
    if args.prior_flags is not None:
        base.prior_flags = args.prior_flags
    if args.customer_tier:
        base.customer_tier = args.customer_tier
    if args.document_id:
        base.document_id = args.document_id
    if args.reviewer_goal:
        base.reviewer_goal = args.reviewer_goal

    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=sorted(SAMPLE_REQUESTS), default="safe")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "live"],
        default="deterministic",
    )
    parser.add_argument(
        "--backend",
        choices=["marie", "openai"],
        default="openai",
        help="LLM backend for live mode.",
    )
    parser.add_argument("--model", help="Optional model override for live mode.")
    parser.add_argument("--document-id")
    parser.add_argument("--reviewer-goal")
    parser.add_argument("--document-text")
    parser.add_argument("--amount", type=float)
    parser.add_argument("--vendor")
    parser.add_argument("--prior-flags", type=int)
    parser.add_argument(
        "--customer-tier",
        choices=["standard", "priority", "regulated"],
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    request = build_request_from_args(args)
    decision = await run_document_review_workflow(
        request,
        mode=args.mode,
        backend=args.backend,
        model=args.model,
    )
    print(json.dumps(decision.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
