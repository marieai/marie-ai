# Dispatch Confirmation Race Fix Work Slices

**Status:** In progress — Slice 01 complete

**Parent plan:** [Dispatch Confirmation Race Fix Implementation Plan](../../plans/2026-07-25-dispatch-confirmation-race-fix.md)

## Objective

Deliver the gateway dispatch-race fix through small, reviewable units. The parent plan is the design source of truth. These slices define implementation ownership, dependencies, test gates, and merge boundaries.

This work covers gateway dispatch admission, scheduler interpretation of gateway results, ambiguous-dispatch settlement, and the attempt fence needed to make late gateway delivery safe. Executor process health and restart policy are outside every slice.

## Slice index

| Slice | Unit of work | Depends on | Delivery boundary |
|---|---|---|---|
| [01](slice-01-freeze-dispatch-race-contract.md) | Freeze the dispatch-race contract | Existing reproducer | Tests and fixtures only |
| [02](slice-02-dispatch-result-and-handle.md) | Add dispatch result and handle types | 01 | New types and unit tests only |
| [03](slice-03-pre-rpc-failure-propagation.md) | Propagate pre-RPC gateway failures | 02 | Job/gateway/networking path |
| [04](slice-04-structured-scheduler-outcomes.md) | Return structured scheduler outcomes | 03 | Scheduler classification and budgets |
| [05](slice-05-monotonic-dispatch-audit.md) | Make dispatch audit monotonic | 04 | Repository transitions only |
| [06](slice-06-dispatch-unknown-settlement.md) | Settle unknown dispatches | 04, 05 | Bounded reconciliation and capacity ownership |
| [07](slice-07-worker-attempt-fence.md) | Fence worker attempt adoption | 05, 06 | Pre-execution attempt and capacity fence |
| [08](slice-08-independent-dispatch-settlement.md) | Settle dispatch results independently | 04, 06 | Dispatch-cycle batching and logs |
| [09](slice-09-gateway-transport-controls.md) | Preserve gateway transport controls | 03 | Dynamic gateway configuration and transport policy |
| [10](slice-10-gateway-e2e-fault-qualification.md) | Qualify gateway faults end to end | 01–09 | Gated mock faults, stress targets, analyzers |
| [11](slice-11-verification-and-operator-docs.md) | Run verification and update operator docs | 01–10 | Full gates, reports, and documentation |

## Delivery sequence

~~~text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07
                    |           |
                    |           +-> 08
                    +--------------> 09

01 through 09 -> 10 -> 11
~~~

Slices 08 and 09 may proceed in parallel after their dependencies pass. Slice 10 is the first slice allowed to claim the concrete gateway behavior is fixed end to end.

## Review rules

- Keep one slice as one review unit unless a dependency must land atomically.
- Start with the slice's focused tests and record the exact result.
- Do not broaden a slice into executor reliability work.
- Preserve the parent plan's lifecycle vocabulary: `send_detached` is the ambiguity boundary; `admitted` is durable desired-state admission, not executor receipt.
- Treat terminal state and capacity release as separate responsibilities:
  - `AttemptLifecycleService` settles durable job and attempt state.
  - The scheduler releases a scheduler-owned ticket.
  - Worker terminal cleanup releases a worker-adopted ticket.
- Do not mark a slice complete when a required PostgreSQL integration test was skipped.
- Update this index and the parent plan if review changes a dependency or acceptance contract.

## Shared completion contract

Every completed slice must include:

- changed production and test files
- focused verification command and result
- any skipped database-backed tests
- observed trace or log changes
- confirmation that no executor-restart scope was added
