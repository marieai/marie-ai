from types import SimpleNamespace as NS

from marie.scheduler.util import (
    frontier_candidate_window,
    frontier_slot_filter,
    is_control_flow_entrypoint,
    ordered_leased_jobs,
)


def wi(jid: str, entrypoint: str):
    return NS(id=jid, data={"metadata": {"on": entrypoint}})


def test_frontier_candidate_window_expands_with_available_slots():
    assert frontier_candidate_window(32, {"a": 0, "b": 0}) == 64
    assert frontier_candidate_window(32, {"a": 4, "b": 2}) == 88
    assert frontier_candidate_window(128, {"a": 2}) == 128


def test_control_flow_entrypoint_excludes_executable_guardrail():
    assert not is_control_flow_entrypoint("guardrail_executor://evaluate")
    assert is_control_flow_entrypoint("branch://control")
    assert not is_control_flow_entrypoint("extract_executor://default")


def test_frontier_slot_filter_allows_control_flow_and_runnable_executors():
    filt = frontier_slot_filter(
        {"extract_executor": 2, "index_executor": 0, "guardrail_executor": 0}
    )

    assert filt(wi("b1", "branch://control"))
    assert not filt(wi("g1", "guardrail_executor://evaluate"))
    assert filt(wi("e1", "extract_executor://default"))
    assert not filt(wi("i1", "index_executor://default"))


def test_ordered_leased_jobs_preserves_planned_order():
    planned = [
        ("exe://A", wi("A0", "exe://A")),
        ("exe://A", wi("A1", "exe://A")),
        ("exe://B", wi("B0", "exe://B")),
    ]

    leased = {"B0", "A0"}

    assert [job.id for _, job in ordered_leased_jobs(planned, leased)] == ["A0", "B0"]
