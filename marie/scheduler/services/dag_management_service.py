import asyncio
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Awaitable
from typing import Any, Callable, Dict, List, Optional

from marie.logging_core.logger import MarieLogger
from marie.query_planner.base import QueryPlan
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState
from marie.scheduler.util import executor_name, is_control_flow_entrypoint
from marie.utils.scheduler_trace import scheduler_trace


class DAGManagementService:
    """
    Service for managing DAG lifecycle, hydration, and synchronization.
    Handles all DAG-related operations including loading from database,
    tracking in memory, and responding to state changes.
    """

    def __init__(
        self,
        repository: JobRepository,
        frontier: MemoryFrontier,
        active_dags: Dict[str, QueryPlan],
        notify_callback: Optional[Callable] = None,
        max_active_dags: int = 0,
        admission_lock: Optional[asyncio.Lock] = None,
        slot_snapshot_provider: Optional[Callable[[], Dict[str, int]]] = None,
        job_cache: Optional[dict[str, WorkInfo]] = None,
        terminal_event_callback: Optional[
            Callable[[str, WorkInfo], Awaitable[None]]
        ] = None,
        resolution_retry_limit: int = 3,
        resolution_retry_delay: float = 1.0,
        resolution_retry_backoff: bool = True,
        resolution_retry_max_delay: float = 30.0,
    ):
        """
        Initialize the DAG management service.

        :param repository: JobRepository for database operations
        :param frontier: MemoryFrontier for in-memory DAG tracking (owned by scheduler)
        :param active_dags: Shared active-DAG read model; this service owns mutations
        :param notify_callback: Callback function to trigger scheduler events
        """
        self.logger = MarieLogger(DAGManagementService.__name__)
        self.repository = repository
        self.frontier = frontier
        self.active_dags = active_dags
        self._notify_callback = notify_callback
        self.max_active_dags = max_active_dags
        self._admission_lock = admission_lock or asyncio.Lock()
        self._slot_snapshot_provider = slot_snapshot_provider or (lambda: {})
        self._job_cache = job_cache if job_cache is not None else {}
        self._terminal_event_callback = terminal_event_callback
        self._dag_resolution_lock = AsyncJobLock()
        self._terminal_dag_states: dict[str, str] = {}
        self._resolution_retry_limit = max(0, resolution_retry_limit)
        self._resolution_retry_delay = max(0.0, resolution_retry_delay)
        self._resolution_retry_backoff = resolution_retry_backoff
        self._resolution_retry_max_delay = max(0.0, resolution_retry_max_delay)

        # Sync task
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    # ==================== DAG Hydration ====================

    @staticmethod
    def _is_schedulable_state(state: Any) -> bool:
        if state is None:
            return True
        if isinstance(state, WorkState):
            return state in (WorkState.CREATED, WorkState.RETRY)
        if isinstance(state, str):
            try:
                return WorkState(state.lower()) in (WorkState.CREATED, WorkState.RETRY)
            except ValueError:
                return False
        return False

    @staticmethod
    def _entrypoint(wi: WorkInfo) -> str:
        metadata = wi.data.get("metadata", {}) if isinstance(wi.data, dict) else {}
        return metadata.get("on", "") if isinstance(metadata, dict) else ""

    def _admission_gate(self, nodes: List[WorkInfo]) -> tuple[bool, set[str]]:
        jobs = {wi.id: wi for wi in nodes}
        dependents: dict[str, list[str]] = defaultdict(list)
        blocked_executors: set[str] = set()

        for wi in nodes:
            for dep in wi.dependencies or []:
                dependents[dep].append(wi.id)

        ready = deque(
            wi.id
            for wi in nodes
            if self._is_schedulable_state(wi.state) and not (wi.dependencies or [])
        )
        if not ready:
            return False, set()

        slots = self._slot_snapshot_provider() or {}
        traversed_control: set[str] = set()
        seen: set[str] = set()

        while ready:
            job_id = ready.popleft()
            if job_id in seen:
                continue
            seen.add(job_id)

            wi = jobs.get(job_id)
            if wi is None or not self._is_schedulable_state(wi.state):
                continue

            entrypoint = self._entrypoint(wi)
            if is_control_flow_entrypoint(entrypoint):
                traversed_control.add(job_id)
                for child_id in dependents.get(job_id, []):
                    child = jobs.get(child_id)
                    if child is None or not self._is_schedulable_state(child.state):
                        continue
                    if all(
                        dep in traversed_control for dep in child.dependencies or []
                    ):
                        ready.append(child_id)
                continue

            executor = executor_name(entrypoint)
            if slots.get(executor, 0) > 0:
                return True, set()
            blocked_executors.add(executor)

        if traversed_control and not blocked_executors:
            return True, set()

        return False, blocked_executors

    async def admit_dag(self, dag_id: str, dag: QueryPlan, *, source: str) -> bool:
        admitted, _ = await self._admit_dag(dag_id, dag, source=source)
        return admitted

    async def _admit_hydrated_dag(
        self, dag_id: str, dag: QueryPlan, nodes: List[WorkInfo], *, source: str
    ) -> tuple[bool, str]:
        return await self._admit_dag(dag_id, dag, nodes=nodes, source=source)

    async def _admit_dag(
        self,
        dag_id: str,
        dag: QueryPlan,
        *,
        source: str,
        nodes: Optional[List[WorkInfo]] = None,
    ) -> tuple[bool, str]:
        async with self._admission_lock:
            if dag_id in self.active_dags:
                return True, "already_active"

            if (
                self.max_active_dags > 0
                and len(self.active_dags) >= self.max_active_dags
            ):
                self.logger.debug(
                    f"Skipping DAG {dag_id} from {source}; "
                    f"active_dags={len(self.active_dags)}/{self.max_active_dags}"
                )
                return False, "active_limit"

            if nodes is not None:
                admissible, blocked_executors = self._admission_gate(nodes)
                if not admissible:
                    blocked_summary = ", ".join(sorted(blocked_executors)) or "none"
                    self.logger.debug(
                        f"Skipping DAG {dag_id} from {source}; "
                        f"no runnable executor path (blocked={blocked_summary})"
                    )
                    return False, "executor_capacity"

            if not await self.repository.mark_dag_as_active(dag_id):
                try:
                    diagnostic = await self.repository.diagnose_dag_activation_failure(
                        dag_id
                    )
                except Exception as diagnostic_error:
                    diagnostic = {
                        "dag_id": dag_id,
                        "reason": "diagnostic_query_failed",
                        "error": repr(diagnostic_error),
                    }
                scheduler_trace(
                    "hydrated_dag_activation_failed",
                    source=source,
                    **diagnostic,
                )
                if diagnostic.get("dag_state") in {
                    "cancelled",
                    "completed",
                    "expired",
                    "failed",
                }:
                    candidate_type = "hydration" if nodes is not None else "admission"
                    self.logger.info(
                        f"Discarded stale {candidate_type} candidate {dag_id} "
                        f"from {source}; "
                        f"dag_state={diagnostic.get('dag_state')} "
                        f"job_states={diagnostic.get('job_states')} "
                        f"blocking_jobs={diagnostic.get('blocking_jobs')} "
                        f"historical_blocking_jobs="
                        f"{diagnostic.get('historical_blocking_jobs')} "
                        f"dag_state_history={diagnostic.get('dag_state_history')}"
                    )
                    return False, "stale_terminal_state"

                self.logger.warning(
                    f"Failed to mark DAG {dag_id} active in database "
                    f"from {source}; leaving it out of active_dags. "
                    f"diagnostic={diagnostic}"
                )
                return False, "db_activation_failed"

            if nodes is not None:
                await self.frontier.add_dag(dag, nodes)
            self.active_dags[dag_id] = dag
            return True, "admitted"

    async def hydrate_single_dag(self, dag_id: str) -> bool:
        """
        Hydrate a specific DAG from the database into the MemoryFrontier.

        :param dag_id: The ID of the DAG to hydrate
        :return: True if DAG was hydrated, False if not found or failed
        """
        try:
            self.logger.debug(f"Hydrating single DAG from DB: {dag_id}")

            # Load DAG and jobs from repository
            serialized_dag, job_rows = await self.repository.load_dag_and_jobs(dag_id)

            if serialized_dag is None:
                self.logger.warning(
                    f"DAG {dag_id} not found in database or not eligible for hydration"
                )
                return False

            # DAG is stored as JSON and the PostgreSQL driver returns it as a dict.
            # Convert to QueryPlan object using Pydantic
            try:
                dag = QueryPlan.model_validate(serialized_dag)
            except Exception as e:
                self.logger.error(f"Failed to parse DAG {dag_id}: {e}")
                traceback.print_exc()
                return False

            # Parse the jobs (also stored as JSON)
            nodes = []
            for _, job_dict in job_rows:
                try:
                    # Manually construct WorkInfo with field mapping and defaults
                    state_raw = job_dict.get("state")
                    wi = WorkInfo(
                        id=str(job_dict["id"]),
                        name=job_dict["name"],
                        priority=job_dict["priority"],
                        state=WorkState(state_raw) if state_raw else None,
                        retry_limit=job_dict["retry_limit"],
                        start_after=job_dict["start_after"],
                        expire_in_seconds=job_dict.get("expire_in_seconds", 0),
                        data=job_dict["data"],
                        retry_delay=job_dict["retry_delay"],
                        retry_backoff=job_dict["retry_backoff"],
                        keep_until=job_dict["keep_until"],
                        dag_id=dag_id,
                        job_level=job_dict["job_level"],
                        soft_sla=job_dict.get("soft_sla"),
                        hard_sla=job_dict.get("hard_sla"),
                    )
                    # Handle dependencies separately
                    deps = job_dict.get("dependencies") or []
                    wi.dependencies = [str(d) for d in deps]
                    nodes.append(wi)
                except Exception as e:
                    self.logger.error(f"Failed to parse job for DAG {dag_id}: {e}")
                    traceback.print_exc()
                    continue

            if not nodes:
                self.logger.warning(f"No jobs found for DAG {dag_id}")
                return False

            admitted, _ = await self._admit_hydrated_dag(
                dag_id, dag, nodes, source="hydrate_single_dag"
            )
            if not admitted:
                return False

            self.logger.info(
                f"Successfully hydrated DAG {dag_id} with {len(nodes)} job(s)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to hydrate DAG {dag_id}: {e}")
            traceback.print_exc()
            return False

    async def hydrate_bulk(
        self,
        dag_batch_size: int = 1000,
        itersize: int = 5000,
        log_every_seconds: float = 2.0,
    ) -> None:
        """
        Rebuild MemoryFrontier from DB in two phases with progress & timing logs:
          1) Stream DAGs that still have unfinished work (created/retry).
          2) In batches of DAG IDs, stream their unfinished jobs with already-filtered deps.
          3) Add once per DAG: self.frontier.add_dag(dag, nodes)

        :param dag_batch_size: Number of DAGs to process in each batch
        :param itersize: Cursor iteration size for streaming
        :param log_every_seconds: How often to log progress
        """

        t0 = time.monotonic()
        self.logger.info("Hydrate: phase 1 (DAG discovery) started...")

        dag_rows = await self.repository.discover_hydratable_dags()
        discover_elapsed = time.monotonic() - t0
        self.logger.info(
            f"Hydrate: phase 1 complete — discovered {len(dag_rows)} DAG(s) in {discover_elapsed:.2f}s "
            f"({(len(dag_rows) / discover_elapsed if discover_elapsed > 0 else 0):.1f} DAGs/sec)."
        )

        # Build map of dag_id -> QueryPlan
        dags: dict[str, QueryPlan] = {}
        dag_ids_ordered: list[str] = []
        parse_skipped = 0
        for dag_id, dag_def in dag_rows:
            if not dag_def:
                parse_skipped += 1
                self.logger.warning(
                    f"Hydrate: DAG {dag_id} has no serialized_dag; skipping."
                )
                continue
            try:
                dags[str(dag_id)] = QueryPlan(**dag_def)
                dag_ids_ordered.append(str(dag_id))
            except Exception as e:
                parse_skipped += 1
                self.logger.error(f"Hydrate: unable to parse DAG {dag_id}: {e}")

        if not dags:
            total_elapsed = time.monotonic() - t0
            self.logger.info(
                f"Hydrate: no DAGs to hydrate (skipped {parse_skipped}). Done in {total_elapsed:.2f}s."
            )
            if self._notify_callback:
                await self._notify_callback()
            return

        self.logger.info(
            f"Hydrate: {len(dags)} DAG(s) ready for job loading "
            f"(skipped {parse_skipped}, total discovered {len(dag_rows)})."
        )

        def _chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        self.logger.info(
            f"Hydrate: phase 2 (job loading) — {len(dag_ids_ordered)} DAG(s), "
            f"batch size {dag_batch_size}, cursor itersize {itersize}."
        )

        buckets: dict[str, list[WorkInfo]] = defaultdict(list)

        # Progress counters
        total_dags = len(dag_ids_ordered)
        processed_dags = 0
        processed_jobs = 0
        last_log_t = time.monotonic()
        phase2_start = last_log_t

        # For batch-level logging
        batch_idx = 0
        for batch in _chunks(dag_ids_ordered, dag_batch_size):
            batch_idx += 1
            b_start = time.monotonic()

            rows = await self.repository.load_hydratable_jobs(batch)

            for dag_id, j in rows:
                dag_id = str(dag_id)
                if dag_id not in dags:
                    continue
                try:
                    state_raw = j.get("state")
                    wi = WorkInfo(
                        id=str(j["id"]),
                        name=j["name"],
                        priority=j["priority"],
                        state=WorkState(state_raw) if state_raw else None,
                        retry_limit=j["retry_limit"],
                        start_after=j["start_after"],
                        expire_in_seconds=0,
                        data=j["data"],
                        retry_delay=j["retry_delay"],
                        retry_backoff=j["retry_backoff"],
                        keep_until=j["keep_until"],
                        dag_id=dag_id,
                        job_level=j["job_level"],
                        soft_sla=j.get("soft_sla"),
                        hard_sla=j.get("hard_sla"),
                    )
                    deps = j.get("dependencies") or []
                    wi.dependencies = [str(d) for d in deps]
                    buckets[dag_id].append(wi)
                    processed_jobs += 1
                except Exception as e:
                    self.logger.error(
                        f"Hydrate: failed to build WorkInfo for DAG {dag_id}: {e}"
                    )

            processed_dags += len(batch)

            # Per-batch timing
            b_elapsed = time.monotonic() - b_start
            self.logger.info(
                f"Hydrate: batch {batch_idx} — {len(batch)} DAG(s), "
                f"{len(rows)} job(s) in {b_elapsed:.2f}s"
            )

            # Progress logging
            now = time.monotonic()
            if now - last_log_t >= log_every_seconds:
                pct = (processed_dags / total_dags) * 100 if total_dags else 0
                elapsed_so_far = now - phase2_start
                self.logger.info(
                    f"Hydrate: progress {processed_dags}/{total_dags} DAGs ({pct:.1f}%), "
                    f"{processed_jobs} jobs, {elapsed_so_far:.2f}s"
                )
                last_log_t = now

        # Phase 3: add DAGs to frontier
        self.logger.info(f"Hydrate: phase 3 (add to frontier) — {len(buckets)} DAG(s)")
        added = 0
        skipped = 0
        deferred_limit = 0
        deferred_capacity = 0
        stale_candidates = 0
        for dag_id in dag_ids_ordered:
            if dag_id not in buckets:
                skipped += 1
                continue
            nodes = buckets[dag_id]
            if not nodes:
                skipped += 1
                continue
            try:
                admitted, reason = await self._admit_hydrated_dag(
                    dag_id, dags[dag_id], nodes, source="hydrate_bulk"
                )
                if admitted:
                    added += 1
                elif reason == "active_limit":
                    deferred_limit += 1
                elif reason == "executor_capacity":
                    deferred_capacity += 1
                elif reason == "stale_terminal_state":
                    stale_candidates += 1
                else:
                    skipped += 1
            except Exception as e:
                self.logger.error(f"Hydrate: frontier.add_dag failed for {dag_id}: {e}")
                skipped += 1

        total_elapsed = time.monotonic() - t0
        self.logger.info(
            f"Hydrate: complete — {added} DAG(s) added to frontier, "
            f"{deferred_limit} deferred by active DAG limit, "
            f"{deferred_capacity} deferred by executor capacity, "
            f"{stale_candidates} stale candidate(s), {skipped} skipped, "
            f"{processed_jobs} job(s) total. "
            f"Total time: {total_elapsed:.2f}s."
        )

        if self._notify_callback:
            await self._notify_callback()

    async def get_dag(self, dag_id: str) -> Optional[QueryPlan]:
        """
        Retrieve a DAG by its ID, using in-memory cache if available.
        Falls back to loading from db if missing.

        :param dag_id: DAG ID
        :return: QueryPlan object if found, None otherwise
        """
        # Return from cache if present
        if dag_id in self.active_dags:
            return self.active_dags[dag_id]

        # Not in memory, try to load from DB
        dag = await self.repository.get_dag_by_id(dag_id)
        if dag:
            self.logger.debug(f"Loaded DAG from DB: {dag_id}")
        else:
            self.logger.warning(f"DAG not found: {dag_id}")

        return dag

    async def evict_dag(
        self,
        dag_id: str,
        reason: str,
        *,
        clear_terminal_state: bool = True,
    ) -> bool:
        if clear_terminal_state:
            self._terminal_dag_states.pop(dag_id, None)

        dag_jobs = await self.frontier.get_jobs_by_dag_id(dag_id)
        stats = await self.frontier.finalize_dag(dag_id)
        for dag_job in dag_jobs:
            self._job_cache.pop(dag_job.id, None)

        removed = self.active_dags.pop(dag_id, None) is not None
        self.logger.info(
            f"Evicted DAG {dag_id} from memory ({reason}); "
            f"removed={removed}, finalize_stats={stats}"
        )
        return removed

    async def reset_all_dags(self) -> Dict[str, Any]:
        """
        Reset the active DAGs dictionary, clearing all currently tracked DAGs.
        This can be useful for debugging or when you need to force a fresh state.

        :return: Dictionary with reset operation details
        """
        try:
            cleared_count = len(self.active_dags) if self.active_dags else 0
            cleared_dags = list(self.active_dags.keys()) if self.active_dags else []

            for dag_id in cleared_dags:
                await self.evict_dag(dag_id, "active DAG reset")

            self.active_dags.clear()

            self.logger.info(f"Reset active DAGs: cleared {cleared_count} DAGs")
            if cleared_dags:
                self.logger.debug(f"Cleared DAGs: {cleared_dags}")

            return {
                "success": True,
                "cleared_count": cleared_count,
                "cleared_dags": cleared_dags,
                "message": f"Successfully reset active DAGs, cleared {cleared_count} DAGs",
            }
        except Exception as e:
            error_msg = f"Failed to reset active DAGs: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    # ==================== DAG State Change Handling ====================

    async def handle_state_change(self, payload: dict) -> None:
        """
        Handle a DAG state change notification from PostgreSQL.

        Optimized payload structure:
        - UPDATE: {'dag_id': '<id>', 'state': '<new_state>', 'op': 'UPDATE'}
        - DELETE: {'dag_id': '<id>', 'op': 'DELETE'}

        :param payload: The notification payload with minimal fields (dag_id, state, op)
        """
        try:
            op = payload.get("op")
            dag_id = payload.get("dag_id")
            if not dag_id:
                self.logger.warning(f"Received notification without dag_id: {payload}")
                return

            if op == "DELETE":
                await self.evict_dag(dag_id, "deleted from database")
            elif op == "UPDATE":
                new_state = payload.get("state")
                if new_state == "created":
                    await self.evict_dag(dag_id, "reset to created")
                    if not await self.hydrate_single_dag(dag_id):
                        self.logger.warning(
                            f"Could not re-hydrate DAG {dag_id}; "
                            "it may not have eligible jobs"
                        )
                elif new_state in {"cancelled", "suspended"}:
                    await self.evict_dag(dag_id, f"state changed to {new_state}")
                elif new_state in {"completed", "failed"}:
                    await self.evict_dag(
                        dag_id,
                        f"state changed to {new_state}",
                        clear_terminal_state=False,
                    )
                elif new_state in {"active", "running", "pending"}:
                    if dag_id not in self.active_dags:
                        self.logger.debug(
                            f"DAG {dag_id} is '{new_state}' in DB but is not local "
                            "to this scheduler yet; it may be admitted by the current "
                            "cycle, owned by another scheduler, or hydrated later."
                        )
                else:
                    self.logger.warning(
                        f"Unknown DAG state '{new_state}' for DAG {dag_id}"
                    )
            else:
                self.logger.warning(f"Unknown operation '{op}' in DAG notification")

            if self._notify_callback:
                await self._notify_callback()
        except Exception as error:
            self.logger.error(
                f"Error handling DAG state notification: {error}", exc_info=True
            )

    def _get_resolution_retry_delay(self, retry_number: int) -> float:
        if not self._resolution_retry_backoff or retry_number <= 1:
            return self._resolution_retry_delay
        return min(
            self._resolution_retry_max_delay,
            self._resolution_retry_delay * (2 ** (retry_number - 1)),
        )

    async def resolve_dag_status_with_retry(
        self,
        job_id: str,
        work_info: WorkInfo,
        *,
        source: str,
    ) -> bool:
        retry_number = 0
        while True:
            try:
                return await self.resolve_dag_status(job_id, work_info)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                retry_number += 1
                if retry_number > self._resolution_retry_limit:
                    self.logger.error(
                        f"Exhausted {self._resolution_retry_limit} DAG resolution "
                        f"retries for dag={work_info.dag_id}, job={job_id}, "
                        f"source={source}: {error}"
                    )
                    return False

                delay = self._get_resolution_retry_delay(retry_number)
                self.logger.warning(
                    f"Retrying DAG resolution {retry_number}/"
                    f"{self._resolution_retry_limit} for dag={work_info.dag_id}, "
                    f"job={job_id}, source={source} after error: {error}; "
                    f"waiting {delay:.2f}s"
                )
                if delay > 0:
                    await asyncio.sleep(delay)

    async def resolve_dag_status(
        self,
        job_id: str,
        work_info: WorkInfo,
    ) -> bool:
        dag_id = work_info.dag_id
        if not dag_id:
            self.logger.warning(
                f"Skipping DAG status resolution for job without dag_id: {job_id}"
            )
            return False

        async with self._dag_resolution_lock[dag_id]:
            dag_state = await self.repository.resolve_dag_state(dag_id)
            if dag_state not in {"completed", "failed"}:
                return False

            previous_state = self._terminal_dag_states.get(dag_id)
            if previous_state is not None:
                self.logger.debug(
                    f"DAG {dag_id} was already handled as {previous_state}"
                )
                return False

            self._terminal_dag_states[dag_id] = dag_state
            try:
                if dag_state == "failed":
                    cancelled = await self.repository.cancel_pending_jobs_for_dag(
                        dag_id=dag_id,
                        output_metadata={
                            "on_complete": "failed",
                            "cancel_reason": "dag_failed",
                            "terminal_dag_state": dag_state,
                            "resolved_by_job_id": job_id,
                        },
                    )
                    self.logger.info(
                        f"Cancelled {cancelled} pending jobs for failed DAG {dag_id}"
                    )

                await self.evict_dag(
                    dag_id,
                    f"resolved as {dag_state}",
                    clear_terminal_state=False,
                )
                if self._terminal_event_callback:
                    await self._terminal_event_callback(dag_state, work_info)
                return True
            except Exception:
                self._terminal_dag_states.pop(dag_id, None)
                raise

    # ==================== DAG Synchronization ====================

    async def start_sync(self, sync_interval: int = 30) -> None:
        if self._sync_task and not self._sync_task.done():
            self.logger.warning("DAG sync task already running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(
            self._sync_loop(sync_interval), name="scheduler-dag-sync"
        )
        self.logger.info(f"Started DAG sync task (interval: {sync_interval}s)")

    async def stop_sync(self) -> None:
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
            self.logger.info("Stopped DAG sync task")

    async def _sync_loop(self, interval: int) -> None:
        scheduler_trace("scheduler_dag_sync_loop_start", interval=interval)

        while self._running:
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                scheduler_trace("scheduler_dag_sync_cycle_failed", error=repr(error))
                self.logger.error(f"Error validating DAGs: {error}")

            await asyncio.sleep(interval)

        scheduler_trace("scheduler_dag_sync_loop_stopped")

    async def sync_once(self) -> None:
        if not self.active_dags:
            scheduler_trace("scheduler_dag_sync_cycle_skipped", reason="no_active_dags")
            self.logger.debug("No active DAGs in memory to validate")
            return

        memory_dag_ids = list(self.active_dags.keys())
        scheduler_trace(
            "scheduler_dag_sync_cycle_start", active_dags=len(memory_dag_ids)
        )
        self.logger.debug(f"Validating {len(memory_dag_ids)} DAGs in memory")

        terminal_dags: set[str] = set()
        for dag_id in memory_dag_ids:
            try:
                dag_state = await self.repository.resolve_dag_state(dag_id)
                if dag_state in {"completed", "failed"}:
                    terminal_dags.add(dag_id)
            except Exception as error:
                self.logger.warning(
                    f"Failed to resolve DAG state for {dag_id} during sync: {error}"
                )

        valid_db_dags = await self.repository.get_active_dag_ids(memory_dag_ids)
        invalid_dags = (set(memory_dag_ids) - valid_db_dags).union(terminal_dags)

        if invalid_dags:
            self.logger.info(f"Found {len(invalid_dags)} invalid DAGs in memory")
            for dag_id in sorted(invalid_dags):
                await self.evict_dag(
                    dag_id,
                    "no longer active or deleted in database",
                    clear_terminal_state=dag_id not in terminal_dags,
                )
            if self._notify_callback:
                await self._notify_callback()
        else:
            self.logger.debug("All DAGs in memory are still valid")

        scheduler_trace(
            "scheduler_dag_sync_cycle_done",
            active_dags=len(memory_dag_ids),
            valid_dags=len(valid_db_dags),
            terminal_dags=len(terminal_dags),
            invalid_dags=len(invalid_dags),
        )

    async def refresh_frontier_priorities(
        self,
        hydrate_missing_limit: int = 100,
        refresh_id: Optional[int] = None,
        source: str = "unknown",
    ) -> Dict[str, int]:
        """
        Refresh manual priorities from DB and hydrate missing DAGs that are
        currently eligible for the frontier.
        """
        tracked_job_ids = list(self.frontier.jobs_by_id.keys())
        changed = 0
        if tracked_job_ids:
            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_frontier_priority_load_start",
                source=source,
                refresh_id=refresh_id,
                tracked=len(tracked_job_ids),
            )
            priorities = await self.repository.get_job_priorities(tracked_job_ids)
            scheduler_trace(
                "scheduler_priority_refresh_frontier_priority_load_done",
                source=source,
                refresh_id=refresh_id,
                tracked=len(tracked_job_ids),
                fetched=len(priorities),
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
            )
            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_frontier_priority_apply_start",
                source=source,
                refresh_id=refresh_id,
                tracked=len(tracked_job_ids),
                fetched=len(priorities),
            )
            changed = await self.frontier.refresh_priorities(priorities)
            scheduler_trace(
                "scheduler_priority_refresh_frontier_priority_apply_done",
                source=source,
                refresh_id=refresh_id,
                tracked=len(tracked_job_ids),
                fetched=len(priorities),
                changed=changed,
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
            )
        else:
            priorities = {}

        phase_started = time.perf_counter()
        scheduler_trace(
            "scheduler_priority_refresh_frontier_discover_start",
            source=source,
            refresh_id=refresh_id,
            hydrate_missing_limit=hydrate_missing_limit,
            active_dags=len(self.active_dags),
            max_active_dags=self.max_active_dags,
        )
        hydratable_dags = await self.repository.discover_hydratable_dags(
            hydrate_missing_limit
        )
        scheduler_trace(
            "scheduler_priority_refresh_frontier_discover_done",
            source=source,
            refresh_id=refresh_id,
            hydrate_missing_limit=hydrate_missing_limit,
            hydratable_dags=len(hydratable_dags),
            active_dags=len(self.active_dags),
            max_active_dags=self.max_active_dags,
            elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
        )

        hydrated = 0
        for index, (dag_id, _serialized_dag) in enumerate(hydratable_dags):
            if dag_id in self.active_dags:
                scheduler_trace(
                    "scheduler_priority_refresh_frontier_hydrate_skip",
                    source=source,
                    refresh_id=refresh_id,
                    dag_id=dag_id,
                    index=index,
                    reason="already_active",
                    active_dags=len(self.active_dags),
                    max_active_dags=self.max_active_dags,
                )
                continue
            if dag_id in self.frontier.dag_nodes:
                scheduler_trace(
                    "scheduler_priority_refresh_frontier_hydrate_skip",
                    source=source,
                    refresh_id=refresh_id,
                    dag_id=dag_id,
                    index=index,
                    reason="frontier_tracked",
                    active_dags=len(self.active_dags),
                    max_active_dags=self.max_active_dags,
                )
                continue
            if (
                self.max_active_dags > 0
                and len(self.active_dags) >= self.max_active_dags
            ):
                scheduler_trace(
                    "scheduler_priority_refresh_frontier_hydrate_stop",
                    source=source,
                    refresh_id=refresh_id,
                    dag_id=dag_id,
                    index=index,
                    reason="active_limit",
                    active_dags=len(self.active_dags),
                    max_active_dags=self.max_active_dags,
                )
                self.logger.debug(
                    f"Skipping DAG hydration refresh at capacity "
                    f"{len(self.active_dags)}/{self.max_active_dags}"
                )
                break
            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_frontier_hydrate_start",
                source=source,
                refresh_id=refresh_id,
                dag_id=dag_id,
                index=index,
                active_dags=len(self.active_dags),
                max_active_dags=self.max_active_dags,
            )
            hydrated_dag = await self.hydrate_single_dag(dag_id)
            scheduler_trace(
                "scheduler_priority_refresh_frontier_hydrate_done",
                source=source,
                refresh_id=refresh_id,
                dag_id=dag_id,
                index=index,
                admitted=hydrated_dag,
                active_dags=len(self.active_dags),
                max_active_dags=self.max_active_dags,
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
            )
            if hydrated_dag:
                hydrated += 1

        if changed > 0 or hydrated > 0:
            self.logger.info(
                f"Refreshed priorities from DB: tracked={len(tracked_job_ids)}, "
                f"fetched={len(priorities)}, changed={changed}, hydrated_missing={hydrated}"
            )

        return {
            "tracked": len(tracked_job_ids),
            "fetched": len(priorities),
            "changed": changed,
            "hydrated_missing": hydrated,
        }

    def get_active_dag_count(self) -> int:
        """Get the count of active DAGs in memory."""
        return len(self.active_dags)

    def get_active_dag_ids(self) -> List[str]:
        """Get list of active DAG IDs."""
        return list(self.active_dags.keys())

    def is_dag_active(self, dag_id: str) -> bool:
        """Check if a DAG is currently active in memory."""
        return dag_id in self.active_dags
