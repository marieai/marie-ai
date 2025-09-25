import asyncio
import random
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Awaitable, Callable, Dict, Union

from marie.helper import get_or_reuse_loop
from marie.logging_core.logger import MarieLogger
from marie.scheduler.models import HeartbeatConfig
from marie.scheduler.printers import (
    print_dag_state_summary,
    print_job_state_summary,
    print_slots_table,
)
from marie.scheduler.repository import SchedulerRepository
from marie.scheduler.util import available_slots_by_executor
from marie.serve.runtimes.servers.cluster_state import ClusterState

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class SchedulerHeartbeat:
    """Handles the heartbeat monitoring for the scheduler."""

    def __init__(
        self,
        scheduler: Any,
        config: SchedulerRepository,
        db_query: SchedulerRepository,
        logger: MarieLogger,
    ):
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self._loop = get_or_reuse_loop()
        self.running = False
        self._task = None
        self._db_query = db_query
        self._db_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hb-executor"
        )

    async def start(self):
        """Starts the heartbeat loop."""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stops the heartbeat loop."""
        if self.running:
            self.running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    self.logger.info("Heartbeat task cancelled.")

    async def _safe_count_states(
        self,
        count_method: Union[
            Callable[[], Dict[str, Any]], Callable[[], Awaitable[Dict[str, Any]]]
        ],
    ) -> Dict[str, Any]:
        """Safely call count methods with proper async handling."""
        try:
            if asyncio.iscoroutinefunction(count_method):
                return await count_method()
            else:
                return await self._loop.run_in_executor(self._db_executor, count_method)
        except Exception as e:
            self.logger.error(f"Error in count method: {e}")
            return {}

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat with throughput, rolling averages, and trend analysis (global + per queue)."""

        self.logger.info(
            f"Heartbeat loop started: interval={self.config.interval}s, "
            f"window={self.config.window_minutes}m, recent={self.config.recent_window_minutes}m"
        )

        _seen_executors = set()
        _max_seen_executors = {}

        last_heartbeat_time = None
        last_completed_jobs = {}
        last_completed_dags = {}
        history = deque()  # (timestamp, jobs_per_queue, dags_per_queue)

        rolling_job_rates = deque(maxlen=self.config.trend_points)
        rolling_dag_rates = deque(maxlen=self.config.trend_points)

        rolling_job_rates_per_queue = defaultdict(
            lambda: deque(maxlen=self.config.trend_points)
        )
        rolling_dag_rates_per_queue = defaultdict(
            lambda: deque(maxlen=self.config.trend_points)
        )

        def _get_completed_per_queue(states: dict, key: str = "completed") -> dict:
            if not states or "queues" not in states:
                return {}
            return {qname: q.get(key, 0) for qname, q in states["queues"].items()}

        def _calc_throughput(curr: dict, prev: dict, minutes: float) -> dict:
            return {
                qname: (
                    max(0.0, (curr[qname] - prev.get(qname, 0)) / minutes)
                    if minutes > 0
                    else 0.0
                )
                for qname in curr
            }

        def _trend(values: deque) -> str:
            """Return colored trend arrow (⬆️, ⬇️, ➡️)."""
            if not self.config.enable_trend_arrows or len(values) < 2:
                return ""
            first, last = values[0], values[-1]
            if last > first * 1.05:
                return f"{GREEN}⬆️{RESET}"
            elif last < first * 0.95:
                return f"{RED}⬇️{RESET}"
            return f"{YELLOW}➡️{RESET}"

        retry_count = 0

        while self.running:
            try:
                unix_now = time.time()
                queue_size = self.scheduler._event_queue.qsize()
                active_dags = list(self.scheduler.active_dags.keys())
                slot_info = None

                # Executor stats collection (configurable)
                if self.config.enable_executor_stats:
                    slot_info = available_slots_by_executor(ClusterState.deployments)
                    _seen_executors.update(slot_info.keys())

                    for executor, count in slot_info.items():
                        current_max = _max_seen_executors.get(executor, 0)
                        _max_seen_executors[executor] = max(current_max, count)

                dag_states = await self._safe_count_states(
                    self._db_query.count_dag_states
                )
                job_states = await self._safe_count_states(
                    self._db_query.count_job_states
                )

                current_completed_jobs = _get_completed_per_queue(
                    job_states, "completed"
                )
                current_completed_dags = _get_completed_per_queue(
                    dag_states, "completed"
                )

                total_completed_jobs = sum(current_completed_jobs.values())
                total_completed_dags = sum(current_completed_dags.values())

                history.append(
                    (
                        unix_now,
                        current_completed_jobs.copy(),
                        current_completed_dags.copy(),
                    )
                )

                # Clean up old history (beyond rolling window)
                cutoff = unix_now - self.config.window_minutes * 60
                while history and history[0][0] < cutoff:
                    history.popleft()

                # --- Recent Throughput ---
                recent_window_seconds = self.config.recent_window_minutes * 60.0
                recent_cutoff = unix_now - recent_window_seconds

                recent_history = [
                    entry for entry in history if entry[0] >= recent_cutoff
                ]

                if len(recent_history) >= 2:
                    # Use oldest and newest entries from recent window
                    t0, jobs0, dags0 = recent_history[0]
                    t1, jobs1, dags1 = recent_history[-1]
                    time_diff_minutes = (t1 - t0) / 60.0

                    if time_diff_minutes > 0:
                        jobs_per_queue_instant = _calc_throughput(
                            jobs1, jobs0, time_diff_minutes
                        )
                        dags_per_queue_instant = _calc_throughput(
                            dags1, dags0, time_diff_minutes
                        )
                        jobs_per_min_global_instant = sum(
                            jobs_per_queue_instant.values()
                        )
                        dags_per_min_global_instant = sum(
                            dags_per_queue_instant.values()
                        )
                    else:
                        jobs_per_queue_instant = {}
                        dags_per_queue_instant = {}
                        jobs_per_min_global_instant = dags_per_min_global_instant = 0.0
                elif last_heartbeat_time is not None:
                    # Fallback to heartbeat-based calculation if insufficient recent history
                    time_diff_minutes = (unix_now - last_heartbeat_time) / 60.0
                    jobs_per_queue_instant = _calc_throughput(
                        current_completed_jobs, last_completed_jobs, time_diff_minutes
                    )
                    dags_per_queue_instant = _calc_throughput(
                        current_completed_dags, last_completed_dags, time_diff_minutes
                    )
                    jobs_per_min_global_instant = sum(jobs_per_queue_instant.values())
                    dags_per_min_global_instant = sum(dags_per_queue_instant.values())
                else:
                    jobs_per_queue_instant = {}
                    dags_per_queue_instant = {}
                    jobs_per_min_global_instant = dags_per_min_global_instant = 0.0

                # --- Rolling throughput (full window) ---
                jobs_per_min_global_window = 0.0
                dags_per_min_global_window = 0.0
                jobs_per_queue_window = defaultdict(float)
                dags_per_queue_window = defaultdict(float)
                window_suffix = " (insufficient data)"

                if len(history) >= 2:
                    # Time-weighted rolling average calculation.
                    # This is more robust against bursty traffic than a simple start/end point calculation.
                    total_job_rate_x_time = 0.0
                    total_dag_rate_x_time = 0.0
                    total_time_diff = 0.0

                    per_queue_job_rate_x_time = defaultdict(float)
                    per_queue_dag_rate_x_time = defaultdict(float)

                    for i in range(1, len(history)):
                        t_prev, jobs_prev, dags_prev = history[i - 1]
                        t_curr, jobs_curr, dags_curr = history[i]

                        interval_minutes = (t_curr - t_prev) / 60.0
                        if interval_minutes <= 0:
                            continue

                        total_time_diff += interval_minutes

                        # Calculate global rate for this interval
                        interval_jobs_completed = sum(jobs_curr.values()) - sum(
                            jobs_prev.values()
                        )
                        interval_dags_completed = sum(dags_curr.values()) - sum(
                            dags_prev.values()
                        )

                        interval_job_rate = (
                            max(0, interval_jobs_completed) / interval_minutes
                        )
                        interval_dag_rate = (
                            max(0, interval_dags_completed) / interval_minutes
                        )

                        total_job_rate_x_time += interval_job_rate * interval_minutes
                        total_dag_rate_x_time += interval_dag_rate * interval_minutes

                        # Calculate per-queue rate for this interval
                        all_queues = set(jobs_curr.keys()) | set(jobs_prev.keys())
                        for q in all_queues:
                            q_jobs_completed = jobs_curr.get(q, 0) - jobs_prev.get(q, 0)
                            q_dags_completed = dags_curr.get(q, 0) - dags_prev.get(q, 0)
                            q_job_rate = max(0, q_jobs_completed) / interval_minutes
                            q_dag_rate = max(0, q_dags_completed) / interval_minutes
                            per_queue_job_rate_x_time[q] += (
                                q_job_rate * interval_minutes
                            )
                            per_queue_dag_rate_x_time[q] += (
                                q_dag_rate * interval_minutes
                            )

                    if total_time_diff > 0:
                        jobs_per_min_global_window = (
                            total_job_rate_x_time / total_time_diff
                        )
                        dags_per_min_global_window = (
                            total_dag_rate_x_time / total_time_diff
                        )
                        for q in per_queue_job_rate_x_time:
                            jobs_per_queue_window[q] = (
                                per_queue_job_rate_x_time[q] / total_time_diff
                            )
                        for q in per_queue_dag_rate_x_time:
                            dags_per_queue_window[q] = (
                                per_queue_dag_rate_x_time[q] / total_time_diff
                            )

                    actual_window_minutes = (history[-1][0] - history[0][0]) / 60.0
                    if actual_window_minutes < self.config.window_minutes * 0.9:
                        window_suffix = f" ({actual_window_minutes:.1f}m data)"
                    else:
                        window_suffix = ""
                else:
                    # Fallback for rolling if not enough history
                    jobs_per_queue_window = jobs_per_queue_instant.copy()
                    dags_per_queue_window = dags_per_queue_instant.copy()
                    jobs_per_min_global_window = jobs_per_min_global_instant
                    dags_per_min_global_window = dags_per_min_global_instant

                rolling_job_rates.append(jobs_per_min_global_window)
                rolling_dag_rates.append(dags_per_min_global_window)

                for q in set(current_completed_jobs) | set(current_completed_dags):
                    rolling_job_rates_per_queue[q].append(
                        jobs_per_queue_window.get(q, 0.0)
                    )
                    rolling_dag_rates_per_queue[q].append(
                        dags_per_queue_window.get(q, 0.0)
                    )

                # Snapshot
                last_heartbeat_time = unix_now
                last_completed_jobs = current_completed_jobs
                last_completed_dags = current_completed_dags

                # --- Logging with configuration ---
                self.logger.info("🔄  Scheduler Heartbeat")
                self.logger.info(f"  📦  Queue Size       : {queue_size}")
                self.logger.info(f"  🧠  Active DAGs      : {len(active_dags)}")
                # Global throughput + trend
                self.logger.info("  📈  Throughput: ")

                self.logger.info(
                    f"  • recent ({self.config.recent_window_minutes}m): {jobs_per_min_global_instant:.2f} jobs/min, "
                    f"{dags_per_min_global_instant:.2f} dags/min"
                )

                trend_job = _trend(rolling_job_rates)
                trend_dag = _trend(rolling_dag_rates)

                self.logger.info(
                    f"  • rolling (last {self.config.window_minutes}m{window_suffix}): "
                    f"{jobs_per_min_global_window:.2f} jobs/min {trend_job}, "
                    f"{dags_per_min_global_window:.2f} dags/min {trend_dag}"
                )
                self.logger.info(
                    f"  ✅  Totals            : {total_completed_jobs} jobs, {total_completed_dags} dags"
                )

                # Per-queue throughput + trend (configurable)
                if self.config.enable_per_queue_stats and (
                    jobs_per_queue_instant or dags_per_queue_instant
                ):
                    self.logger.info("  📊 Per-Queue Throughput:")
                    for qname in sorted(
                        set(current_completed_jobs) | set(current_completed_dags)
                    ):
                        jpm_i = jobs_per_queue_instant.get(qname, 0.0)
                        dpm_i = dags_per_queue_instant.get(qname, 0.0)
                        jpm_w = jobs_per_queue_window.get(qname, 0.0)
                        dpm_w = dags_per_queue_window.get(qname, 0.0)

                        jtot = current_completed_jobs.get(qname, 0)
                        dtot = current_completed_dags.get(qname, 0)

                        jtrend = _trend(rolling_job_rates_per_queue[qname])
                        dtrend = _trend(rolling_dag_rates_per_queue[qname])

                        self.logger.info(
                            f"   • {qname:<12} | Jobs: {jpm_i:.2f}/min ({self.config.recent_window_minutes}m), "
                            f"{jpm_w:.2f}/min ({self.config.window_minutes}m{window_suffix}) {jtrend}"
                        )
                        self.logger.info(
                            f"     {'':<12} | DAGs: {dpm_i:.2f}/min ({self.config.recent_window_minutes}m), "
                            f"{dpm_w:.2f}/min ({self.config.window_minutes}m{window_suffix}) {dtrend}"
                        )
                        self.logger.info(
                            f"     {'':<12} | Totals: {jtot} jobs, {dtot} dags"
                        )

                if self.config.log_active_dags and active_dags:
                    shown = ", ".join(active_dags[:5])
                    suffix = "..." if len(active_dags) > 5 else ""
                    self.logger.debug(f"     DAG IDs          : [{shown}{suffix}]")

                print_dag_state_summary(dag_states)
                print_job_state_summary(job_states)

                if self.config.enable_executor_stats:
                    print_slots_table(slot_info, _max_seen_executors)

                (
                    await self.scheduler.diagnose_pool()
                    if asyncio.iscoroutinefunction(self.scheduler.diagnose_pool)
                    else self.scheduler.diagnose_pool()
                )

                print("Memory Frontier State:")
                frontier_summary = self.scheduler.frontier.summary(detail=True)
                pprint(frontier_summary)

                await asyncio.sleep(self.config.interval)
                retry_count = 0

            except asyncio.CancelledError:
                self.logger.info("Heartbeat loop cancelled.")
                break
            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"Heartbeat loop error (attempt {retry_count}/{self.config.max_retries}): {e}"
                )

                if retry_count >= self.config.max_retries:
                    self.logger.critical(
                        f"Heartbeat loop failed {self.config.max_retries} times, stopping"
                    )
                    break

                backoff_time = self.config.error_backoff * (
                    2 ** (retry_count - 1)
                ) + random.uniform(0, 1)
                self.logger.warning(f"Retrying heartbeat in {backoff_time:.2f}s...")
                await asyncio.sleep(backoff_time)
