# Job parameter and admission boundaries

The scheduler manages DAGs, dependencies, queues, execution attempts and leases. Executor parameters are application data. Adding an executor parameter does not require a scheduler or dispatcher change.

## Transport

The dispatcher forwards application metadata at the same parameter names, including nested values, empty collections, false and zero. The existing `parameters.payload` representation remains available for older executors.

Declared asset fields use the existing asset decoder. It materializes inline bytes into an asset reference and supplies the canonical URI. Inline file bytes are consumed by that decoder. A request without asset fields carries parameters and an empty document list. This decision is independent of executor names and workflow names.

`job_id`, `payload`, `dag_id`, `node_task_id`, `partition_key`, `run_owner`, `run_attempt_id` , `submission_trace_context` and `__results__` are runtime-owned parameter names. Application fields cannot override the execution identity supplied by the scheduler envelope. Per-executor parameter overrides preserve these runtime values; application input cannot preseed the runtime response map.

An executor validates its own required inputs. Invalid input must raise or return the existing `status: error` response so worker completion records a failure. A dictionary containing only an `error` field is not a failure signal.

## Application integration

Application startup may inject a preparation callback and a persistence callback into the scheduler. Preparation runs before planning. Persistence receives the current database transaction after the new DAG and jobs are inserted; a failure rolls the whole transaction back. Conflicting DAG submissions do not execute persistence callbacks. Scheduler admission and notification happen after commit.

The KB implementation lives in `marie/kb/submission.py`. It resolves registered indexing parameters and binds waiting items to the submitted source DAG. The generic scheduler services and repository do not import KB code, query KB tables or select KB workflows.
