import asyncio
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2

from marie.logging_core.logger import MarieLogger
from marie.query_planner.base import QueryPlan
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState


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
        loop: Optional[asyncio.AbstractEventLoop] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        notify_callback: Optional[callable] = None,
    ):
        """
        Initialize the DAG management service.

        :param repository: JobRepository for database operations
        :param frontier: MemoryFrontier for in-memory DAG tracking (owned by scheduler)
        :param active_dags: Active DAGs dictionary (owned by scheduler)
        :param loop: Event loop for async operations
        :param executor: Thread pool executor for blocking operations
        :param notify_callback: Callback function to trigger scheduler events
        """
        self.logger = MarieLogger(DAGManagementService.__name__)
        self.repository = repository
        self.frontier = frontier
        self.active_dags = active_dags  # Reference to scheduler's active_dags
        self._loop = loop or asyncio.get_event_loop()
        self._executor = executor
        self._notify_callback = notify_callback

        # Sync task
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    # ==================== DAG Hydration ====================

    async def hydrate_single_dag(self, dag_id: str) -> bool:
        """
        Hydrate a specific DAG from the database into the MemoryFrontier.

        :param dag_id: The ID of the DAG to hydrate
        :return: True if DAG was hydrated, False if not found or failed
        """
        try:
            self.logger.info(f"Hydrating single DAG from DB: {dag_id}")

            # Load DAG and jobs from repository
            serialized_dag, job_rows = await self.repository.load_dag_and_jobs(dag_id)

            if serialized_dag is None:
                self.logger.warning(
                    f"DAG {dag_id} not found in database or not eligible for hydration"
                )
                return False

            # DAG is stored as JSON, psycopg2 returns it as dict
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

            # Add to frontier
            await self.frontier.add_dag(dag, nodes)

            # Track as active
            self.active_dags[dag_id] = dag

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

        def _stream_dags():
            conn = self.repository._get_connection()
            cur = None
            try:
                cur = conn.cursor(name="hydrate_frontier_dags")
                cur.itersize = itersize
                cur.execute(
                    "SELECT dag_id, serialized_dag FROM marie_scheduler.hydrate_frontier_dags()"
                )
                for dag_id, serialized_dag in cur:
                    yield dag_id, serialized_dag
                if cur and not cur.closed:
                    self.repository._close_cursor(cur)
                cur = None  # so we don't try to close again in finally
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                if cur and not cur.closed:
                    self.repository._close_cursor(cur)
                self.repository._close_connection(conn)

        t0 = time.monotonic()
        self.logger.info("Hydrate: phase 1 (DAG discovery) started…")

        dag_rows = await self._loop.run_in_executor(
            self._executor, lambda: list(_stream_dags())
        )
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

        def _stream_jobs_for_batch(dag_ids_batch):
            conn = self.repository._get_connection()
            cur = None
            try:
                dag_ids_text = [str(x) for x in dag_ids_batch]
                cur = conn.cursor(name="hydrate_frontier_jobs")
                cur.itersize = itersize
                cur.execute(
                    "SELECT dag_id, job FROM marie_scheduler.hydrate_frontier_jobs((%s)::uuid[])",
                    (dag_ids_text,),
                )
                for row in cur:
                    yield row
                if cur and not cur.closed:
                    self.repository._close_cursor(cur)
                cur = None
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                if cur and not cur.closed:
                    self.repository._close_cursor(cur)
                self.repository._close_connection(conn)

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

            rows = await self._loop.run_in_executor(
                self._executor, lambda: list(_stream_jobs_for_batch(batch))
            )

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
        for dag_id in dag_ids_ordered:
            if dag_id not in buckets:
                skipped += 1
                continue
            nodes = buckets[dag_id]
            if not nodes:
                skipped += 1
                continue
            try:
                await self.frontier.add_dag(dags[dag_id], nodes)
                self.active_dags[dag_id] = dags[dag_id]
                added += 1
            except Exception as e:
                self.logger.error(f"Hydrate: frontier.add_dag failed for {dag_id}: {e}")
                skipped += 1

        total_elapsed = time.monotonic() - t0
        self.logger.info(
            f"Hydrate: complete — {added} DAG(s) added to frontier, "
            f"{skipped} skipped, {processed_jobs} job(s) total. "
            f"Total time: {total_elapsed:.2f}s."
        )

        if self._notify_callback:
            await self._notify_callback()

    # ==================== DAG State Management ====================

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

    def remove_dag(self, dag_id: str, reason: str) -> bool:
        """
        Remove a DAG from memory tracking.

        :param dag_id: DAG ID to remove
        :param reason: Reason for removal (for logging)
        :return: True if removed, False if not found
        """
        if dag_id in self.active_dags:
            del self.active_dags[dag_id]
            self.logger.warning(f"Removed DAG {dag_id} from active_dags ({reason})")
            return True
        else:
            self.logger.debug(f"DAG {dag_id} not in active_dags ({reason})")
            return False

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
                await self.frontier.finalize_dag(dag_id)

            self.active_dags = {}

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

            self.logger.info(
                f"Received DAG state notification: op={op}, dag_id={dag_id}"
            )

            if op == "DELETE":
                # DAG was deleted from database, remove from MemoryFrontier
                self.logger.info(
                    f"DAG {dag_id} was deleted, removing from memory frontier"
                )
                stats = await self.frontier.finalize_dag(dag_id)
                self.logger.info(f"Finalized DAG {dag_id} from memory: {stats}")

                # Remove from active_dags tracking
                if dag_id in self.active_dags:
                    del self.active_dags[dag_id]
                    self.logger.info(f"Removed DAG {dag_id} from active_dags")

            elif op == "UPDATE":
                new_state = payload.get("state")
                self.logger.info(f"DAG {dag_id} state changed to: {new_state}")

                # Handle different states appropriately
                if new_state == "created":
                    # DAG was reset (via reset_all or similar)
                    # Remove from memory and re-hydrate from DB
                    self.logger.info(
                        f"DAG {dag_id} reset to 'created' - removing from memory and re-hydrating from DB"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Removed DAG {dag_id} from memory frontier: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(f"Removed DAG {dag_id} from active_dags")

                    # Re-hydrate the DAG from DB with fresh state
                    hydrated = await self.hydrate_single_dag(dag_id)
                    if hydrated:
                        self.logger.info(
                            f"Successfully re-hydrated DAG {dag_id} from database"
                        )
                    else:
                        self.logger.warning(
                            f"Could not re-hydrate DAG {dag_id} - may not have eligible jobs"
                        )

                elif new_state == "cancelled":
                    # DAG was cancelled - remove from active processing
                    self.logger.info(
                        f"DAG {dag_id} cancelled - removing from memory and active processing"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Removed cancelled DAG {dag_id} from memory: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(
                            f"Removed cancelled DAG {dag_id} from active_dags"
                        )

                elif new_state == "suspended":
                    # DAG was suspended - remove from active execution but keep tracked
                    self.logger.info(
                        f"DAG {dag_id} suspended - removing from active execution"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Removed suspended DAG {dag_id} from memory: {stats}"
                    )

                    # Keep in active_dags for tracking but it won't be scheduled
                    # When resumed, it will be re-hydrated from DB

                elif new_state in ["completed", "failed"]:
                    # DAG finished - clean up memory
                    self.logger.info(
                        f"DAG {dag_id} finished with state '{new_state}' - cleaning up memory"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Cleaned up finished DAG {dag_id} from memory: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(
                            f"Removed finished DAG {dag_id} from active_dags"
                        )

                elif new_state in ["running", "pending"]:
                    # DAG should be active - if not in memory, it might need hydration
                    if dag_id not in self.active_dags:
                        self.logger.warning(
                            f"DAG {dag_id} is in '{new_state}' state but not in active_dags. "
                            "It will be hydrated on next scheduler cycle."
                        )

                else:
                    self.logger.warning(
                        f"Unknown DAG state '{new_state}' for DAG {dag_id}"
                    )

            else:
                self.logger.warning(f"Unknown operation '{op}' in DAG notification")

            # Notify scheduler of state change
            if self._notify_callback:
                await self._notify_callback()

        except Exception as e:
            self.logger.error(f"Error handling DAG state notification: {e}")
            traceback.print_exc()

    # ==================== DAG Synchronization ====================

    async def start_sync(self, sync_interval: int = 30):
        """
        Start periodic DAG synchronization task.

        :param sync_interval: How often to sync DAGs (in seconds)
        """
        if self._sync_task:
            self.logger.warning("DAG sync task already running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop(sync_interval))
        self.logger.info(f"Started DAG sync task (interval: {sync_interval}s)")

    async def stop_sync(self):
        """Stop the periodic DAG synchronization task."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
            self.logger.info("Stopped DAG sync task")

    async def _sync_loop(self, interval: int):
        """
        Periodic sync loop that validates DAGs in memory against database.

        :param interval: Sync interval in seconds
        """
        self.logger.info(f"Starting DAG sync loop (interval: {interval}s)")

        while self._running:
            try:
                await self._sync_once()
            except Exception as e:
                self.logger.error(f"Error in DAG sync loop: {e}")

            await asyncio.sleep(interval)

        self.logger.info("DAG sync loop stopped")

    async def _sync_once(self):
        """
        Perform a single synchronization: validate that DAGs in memory
        still exist and are active in the database.
        """
        if not self.active_dags:
            self.logger.debug("No active DAGs in memory to validate")
            return

        memory_dag_ids = list(self.active_dags.keys())
        self.logger.debug(f"Validating {len(memory_dag_ids)} DAGs in memory")

        def _db_check():
            """Synchronous database check."""
            cursor = None
            conn = None
            try:
                placeholders = ",".join(["%s"] * len(memory_dag_ids))
                query = f"""
                    SELECT id FROM marie_scheduler.dag
                    WHERE id IN ({placeholders}) AND state = 'active'
                """

                conn = self.repository._get_connection()
                cursor = self.repository._execute_sql_gracefully(
                    query, memory_dag_ids, return_cursor=True, connection=conn
                )
                if not cursor:
                    return set()

                valid_dag_records = cursor.fetchall()
                return {record[0] for record in valid_dag_records}
            finally:
                self.repository._close_cursor(cursor)
                self.repository._close_connection(conn)

        # Run DB check in executor
        valid_db_dags = await self._loop.run_in_executor(self._executor, _db_check)
        invalid_dags = set(memory_dag_ids) - valid_db_dags

        if invalid_dags:
            self.logger.info(f"Found {len(invalid_dags)} invalid DAGs in memory")
            for dag_id in invalid_dags:
                self.remove_dag(dag_id, "no longer active or deleted in database")
        else:
            self.logger.debug("All DAGs in memory are still valid")

    def get_active_dag_count(self) -> int:
        """Get the count of active DAGs in memory."""
        return len(self.active_dags)

    def get_active_dag_ids(self) -> List[str]:
        """Get list of active DAG IDs."""
        return list(self.active_dags.keys())

    def is_dag_active(self, dag_id: str) -> bool:
        """Check if a DAG is currently active in memory."""
        return dag_id in self.active_dags
