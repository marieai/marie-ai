#!/usr/bin/env python3
"""
Comprehensive stress tester for state management components:
  - SemaphoreStore (reserve, release, release_owned, renew)
  - SlotCapacityManager (capacity reconciliation, node discovery)
  - SemaphoreHolder (ownership validation, lease management)

Tests realistic concurrent scenarios including:
  - Multiple workers competing for slots
  - Dynamic capacity changes
  - Owner validation and security
  - Lease expiration and renewal
  - Counter reconciliation and recovery
"""

import argparse
import asyncio
import logging
import random
import re
import socket
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Set

from marie.logging_core.logger import MarieLogger
from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.semaphore_store import SemaphoreHolder, SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StateManagementStresser")

Mode = Literal[
    "reserve_release",  # Basic reserve → release cycle
    "reserve_release_owned",  # Reserve → release with ownership validation
    "renew_stress",  # Reserve → continuous renewal → release
    "capacity_changes",  # Dynamic capacity changes during operations
    "owner_validation",  # Test ownership security (wrong owner attempts)
    "counter_recovery",  # Simulate counter corruption and recovery
    "slot_manager",  # Test SlotCapacityManager integration
    "full_integration",  # Comprehensive integration test
]


@dataclass
class WorkerMetrics:
    """Metrics tracked per worker"""

    worker_id: str
    reserves_success: int = 0
    reserves_failed: int = 0
    releases_success: int = 0
    releases_failed: int = 0
    renewals_success: int = 0
    renewals_failed: int = 0
    ownership_violations: int = 0
    active_tickets: Set[str] = field(default_factory=set)


@dataclass
class StressMetrics:
    """Overall stress test metrics"""

    total_ops: int = 0
    success_ops: int = 0
    failed_ops: int = 0
    reserve_success: int = 0
    reserve_failed: int = 0
    release_success: int = 0
    release_failed: int = 0
    renew_success: int = 0
    renew_failed: int = 0
    ownership_violations: int = 0
    counter_mismatches: int = 0
    counter_mismatches_transient: int = 0  # Fixed by itself within 100ms
    counter_mismatches_persistent: int = 0  # Still there after 100ms
    reconciliations: int = 0
    capacity_changes: int = 0
    lease_expirations_detected: int = 0
    over_capacity_events: int = 0


class StateManagementStresser:
    """
    Comprehensive stress tester for state management components.

    Tests include:
    - Concurrent reserve/release operations
    - Ownership validation and security
    - Lease renewal under load
    - Dynamic capacity changes
    - Counter reconciliation and recovery
    - SlotCapacityManager integration
    """

    def __init__(
        self,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        mode: Mode = "full_integration",
        num_slots: int = 3,
        num_workers: int = 10,
        capacity_per_slot: int = 20,
        lease_ttl: int = 30,
        enable_chaos: bool = False,
    ):
        self._client = EtcdClient(etcd_host=etcd_host, etcd_port=etcd_port)
        self._sema = SemaphoreStore(self._client, default_lease_ttl=lease_ttl)

        # Create SlotCapacityManager
        marie_logger = MarieLogger("StateStresser")
        self._capacity_mgr = SlotCapacityManager(
            semaphore_store=self._sema,
            logger=marie_logger,
        )

        self.mode: Mode = mode
        self.num_slots = num_slots
        self.num_workers = num_workers
        self.capacity_per_slot = capacity_per_slot
        self.lease_ttl = lease_ttl
        self.enable_chaos = enable_chaos
        self.running = True

        # Create slot types
        self.slot_types = [f"slot_{i}" for i in range(num_slots)]

        # Metrics
        self.metrics = StressMetrics()
        self.worker_metrics: Dict[str, WorkerMetrics] = {}

        # Active tickets tracking
        self._active_tickets: Dict[str, Dict[str, str]] = defaultdict(
            dict
        )  # slot -> {ticket_id: owner}
        self._tickets_lock = asyncio.Lock()

        logger.info(
            f"StateManagementStresser initialized:\n"
            f"  etcd: {etcd_host}:{etcd_port}\n"
            f"  mode: {self.mode}\n"
            f"  slots: {num_slots} x {capacity_per_slot} capacity\n"
            f"  workers: {num_workers}\n"
            f"  lease_ttl: {lease_ttl}s\n"
            f"  chaos: {enable_chaos}"
        )

    @staticmethod
    def _parse_duration(duration_str: str) -> timedelta:
        """Parse duration string like '30s', '2m', '1h'"""
        if not duration_str:
            return timedelta()
        m = re.match(r"(\d+)\s*([hms])?$", duration_str.lower().strip())
        if not m:
            raise ValueError("Invalid duration. Use '30s', '2m', or '1h'")
        value = int(m.group(1))
        unit = m.group(2) or "s"
        if unit == "h":
            return timedelta(hours=value)
        elif unit == "m":
            return timedelta(minutes=value)
        return timedelta(seconds=value)

    def _init_capacities(self):
        """Initialize capacities for all slot types"""
        for slot in self.slot_types:
            self._sema.set_capacity(slot, self.capacity_per_slot)
            logger.info(f"Set capacity for {slot} = {self.capacity_per_slot}")

    def _get_worker_metrics(self, worker_id: str) -> WorkerMetrics:
        """Get or create worker metrics"""
        if worker_id not in self.worker_metrics:
            self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
        return self.worker_metrics[worker_id]

    # ==================== Basic Operations ====================

    def _op_reserve_release(self, worker_id: str, idx: int) -> bool:
        """Basic reserve → release cycle"""
        slot = random.choice(self.slot_types)
        ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
        owner = worker_id

        wm = self._get_worker_metrics(worker_id)

        try:
            # Reserve with retry on CAS contention (not capacity exhaustion)
            max_attempts = 3
            ok = False
            for attempt in range(max_attempts):
                ok = self._sema.reserve(
                    slot_type=slot,
                    ticket_id=ticket_id,
                    node=f"node-{worker_id}",
                    owner=owner,
                )

                if ok:
                    break

                # Check if it's capacity exhaustion vs CAS contention
                avail = self._sema.available_slot_count(slot)
                if avail <= 0:
                    # No capacity, don't retry
                    break

                # CAS contention, brief backoff
                if attempt < max_attempts - 1:
                    time.sleep(0.001 * (attempt + 1))

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1
            wm.active_tickets.add(ticket_id)

            # Simulate work (reduced to 0.1-1ms for higher throughput)
            time.sleep(random.uniform(0.0001, 0.001))

            # Release
            ok_rel = self._sema.release(slot, ticket_id)
            wm.active_tickets.discard(ticket_id)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} reserve_release error: {e}")
            wm.active_tickets.discard(ticket_id)
            return False

    def _op_reserve_release_owned(self, worker_id: str, idx: int) -> bool:
        """Reserve → release with ownership validation"""
        slot = random.choice(self.slot_types)
        ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
        owner = worker_id

        wm = self._get_worker_metrics(worker_id)

        try:
            # Reserve with owner
            ok = self._sema.reserve(
                slot_type=slot,
                ticket_id=ticket_id,
                node=f"node-{worker_id}",
                owner=owner,
            )

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1
            wm.active_tickets.add(ticket_id)

            # Verify holder has correct owner
            holder = self._sema.get_holder(slot, ticket_id)
            if holder and holder.owner != owner:
                logger.error(
                    f"Ownership mismatch! Expected {owner}, got {holder.owner}"
                )
                wm.ownership_violations += 1
                self.metrics.ownership_violations += 1

            # Simulate work (reduced to 0.1-1ms for higher throughput)
            time.sleep(random.uniform(0.0001, 0.001))

            # Release with ownership check
            ok_rel = self._sema.release_owned(slot, ticket_id, owner=owner)
            wm.active_tickets.discard(ticket_id)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} reserve_release_owned error: {e}")
            wm.active_tickets.discard(ticket_id)
            return False

    def _op_renew_stress(self, worker_id: str, idx: int) -> bool:
        """Reserve → continuous renewal → release"""
        slot = random.choice(self.slot_types)
        ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
        owner = worker_id

        wm = self._get_worker_metrics(worker_id)

        try:
            # Reserve with configurable TTL
            ok = self._sema.reserve(
                slot_type=slot,
                ticket_id=ticket_id,
                node=f"node-{worker_id}",
                owner=owner,
                ttl=self.lease_ttl,
            )

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1
            wm.active_tickets.add(ticket_id)

            # Renew multiple times
            num_renewals = random.randint(2, 5)
            renewal_failed = False
            for renewal_num in range(num_renewals):
                time.sleep(random.uniform(0.5, 1.5))
                ok_renew = self._sema.renew(
                    slot,
                    ticket_id,
                    owner=owner,
                    ttl=self.lease_ttl,
                    update_ttl_field=True,
                )
                if ok_renew:
                    wm.renewals_success += 1
                    self.metrics.renew_success += 1
                else:
                    wm.renewals_failed += 1
                    self.metrics.renew_failed += 1
                    renewal_failed = True
                    # CRITICAL: Abort immediately when renewal fails to prevent lease expiration
                    # If we continue, the lease will expire, etcd will auto-delete the holder,
                    # but the counter won't be decremented (because release() will fail with no holder)
                    logger.warning(
                        f"Worker {worker_id} renewal #{renewal_num+1} failed for {ticket_id} "
                        f"on {slot}, aborting job to prevent lease expiration"
                    )
                    break

            # Release
            ok_rel = self._sema.release_owned(slot, ticket_id, owner=owner)
            wm.active_tickets.discard(ticket_id)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                # Log why release failed - holder might have expired or CAS failed
                holder = self._sema.get_holder(slot, ticket_id)
                if holder is None:
                    logger.warning(
                        f"Worker {worker_id} release_owned failed for {ticket_id} on {slot}: "
                        f"holder not found (possibly expired lease)"
                    )
                else:
                    logger.warning(
                        f"Worker {worker_id} release_owned failed for {ticket_id} on {slot}: "
                        f"holder exists but CAS failed or owner mismatch"
                    )
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} renew_stress error: {e}")
            wm.active_tickets.discard(ticket_id)
            return False

    def _op_owner_validation(self, worker_id: str, idx: int) -> bool:
        """Test ownership security - try to release with wrong owner"""
        slot = random.choice(self.slot_types)
        ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
        correct_owner = worker_id
        wrong_owner = f"attacker-{random.randint(1, 100)}"

        wm = self._get_worker_metrics(worker_id)

        try:
            # Reserve with correct owner
            ok = self._sema.reserve(
                slot_type=slot,
                ticket_id=ticket_id,
                node=f"node-{worker_id}",
                owner=correct_owner,
            )

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1

            # Try to release with WRONG owner - should fail
            ok_wrong = self._sema.release_owned(slot, ticket_id, owner=wrong_owner)

            if ok_wrong:
                # This is a security violation!
                logger.error(
                    f"SECURITY VIOLATION: Wrong owner {wrong_owner} released ticket "
                    f"owned by {correct_owner}!"
                )
                wm.ownership_violations += 1
                self.metrics.ownership_violations += 1
                return False

            # Now release with correct owner - should succeed
            ok_rel = self._sema.release_owned(slot, ticket_id, owner=correct_owner)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} owner_validation error: {e}")
            return False

    def _op_counter_recovery(self, worker_id: str, idx: int) -> bool:
        """Simulate counter corruption and recovery"""
        slot = random.choice(self.slot_types)

        try:
            # Sometimes delete the counter to simulate Bug #2
            if random.random() < 0.1:  # 10% chance
                cnt_key = f"semaphores/{slot}/count"
                try:
                    self._client.delete(cnt_key)
                    logger.info(f"[CHAOS] Deleted counter for {slot}")
                except Exception:
                    pass

            # Try normal reserve/release cycle
            ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
            owner = worker_id

            wm = self._get_worker_metrics(worker_id)

            ok = self._sema.reserve(
                slot_type=slot,
                ticket_id=ticket_id,
                node=f"node-{worker_id}",
                owner=owner,
            )

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1

            time.sleep(random.uniform(0.001, 0.005))

            # Release should handle missing counter gracefully (Bug #2 fix)
            ok_rel = self._sema.release_owned(slot, ticket_id, owner=owner)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} counter_recovery error: {e}")
            return False

    def _op_capacity_changes(self, worker_id: str, idx: int) -> bool:
        """Test operations during dynamic capacity changes"""
        slot = random.choice(self.slot_types)
        ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
        owner = worker_id

        wm = self._get_worker_metrics(worker_id)

        try:
            # Randomly change capacity
            if random.random() < 0.05:  # 5% chance
                new_cap = max(1, self.capacity_per_slot + random.randint(-2, 3))
                self._sema.set_capacity(slot, new_cap)
                self.metrics.capacity_changes += 1
                logger.info(f"[CAPACITY] Changed {slot} capacity to {new_cap}")

            # Normal reserve/release
            ok = self._sema.reserve(
                slot_type=slot,
                ticket_id=ticket_id,
                node=f"node-{worker_id}",
                owner=owner,
            )

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1

            time.sleep(random.uniform(0.001, 0.01))

            ok_rel = self._sema.release_owned(slot, ticket_id, owner=owner)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} capacity_changes error: {e}")
            return False

    def _op_slot_manager(self, worker_id: str, idx: int) -> bool:
        """Test SlotCapacityManager integration"""
        try:
            # Simulate node discovery
            if random.random() < 0.1:  # 10% chance
                deployment_nodes = {}
                for slot in self.slot_types:
                    # Simulate discovered nodes for this slot
                    num_nodes = random.randint(1, 3)
                    deployment_nodes[slot] = [
                        {"address": f"192.168.1.{i}", "gateway": "gw"}
                        for i in range(num_nodes)
                    ]

                # Refresh capacities based on discovered nodes
                try:
                    rows, totals, summary = self._capacity_mgr.refresh_from_nodes(
                        deployment_nodes
                    )
                    logger.info(summary)
                except Exception as e:
                    logger.warning(f"[SLOT_MGR] Refresh failed: {e}")

            # Normal reserve/release operation
            slot = random.choice(self.slot_types)
            ticket_id = f"w{worker_id}-t{idx}-{uuid.uuid4().hex[:8]}"
            owner = worker_id

            wm = self._get_worker_metrics(worker_id)

            ok = self._sema.reserve(
                slot_type=slot,
                ticket_id=ticket_id,
                node=f"node-{worker_id}",
                owner=owner,
            )

            if not ok:
                wm.reserves_failed += 1
                self.metrics.reserve_failed += 1
                return False

            wm.reserves_success += 1
            self.metrics.reserve_success += 1

            time.sleep(random.uniform(0.0001, 0.001))

            ok_rel = self._sema.release_owned(slot, ticket_id, owner=owner)

            if ok_rel:
                wm.releases_success += 1
                self.metrics.release_success += 1
                return True
            else:
                wm.releases_failed += 1
                self.metrics.release_failed += 1
                return False

        except Exception as e:
            logger.error(f"Worker {worker_id} slot_manager error: {e}")
            return False

    def _op_full_integration(self, worker_id: str, idx: int) -> bool:
        """Comprehensive integration test - mix of all operations"""
        # Randomly choose an operation type
        op_type = random.choice(
            [
                "reserve_release_owned",
                "renew_stress",
                "owner_validation",
                "capacity_changes",
            ]
        )

        if op_type == "reserve_release_owned":
            return self._op_reserve_release_owned(worker_id, idx)
        elif op_type == "renew_stress":
            return self._op_renew_stress(worker_id, idx)
        elif op_type == "owner_validation":
            return self._op_owner_validation(worker_id, idx)
        elif op_type == "capacity_changes":
            return self._op_capacity_changes(worker_id, idx)
        return False

    def _perform_one(self, worker_id: str, idx: int) -> bool:
        """Execute one operation based on mode"""
        self.metrics.total_ops += 1

        if self.mode == "reserve_release":
            result = self._op_reserve_release(worker_id, idx)
        elif self.mode == "reserve_release_owned":
            result = self._op_reserve_release_owned(worker_id, idx)
        elif self.mode == "renew_stress":
            result = self._op_renew_stress(worker_id, idx)
        elif self.mode == "capacity_changes":
            result = self._op_capacity_changes(worker_id, idx)
        elif self.mode == "owner_validation":
            result = self._op_owner_validation(worker_id, idx)
        elif self.mode == "counter_recovery":
            result = self._op_counter_recovery(worker_id, idx)
        elif self.mode == "slot_manager":
            result = self._op_slot_manager(worker_id, idx)
        elif self.mode == "full_integration":
            result = self._op_full_integration(worker_id, idx)
        else:
            logger.error(f"Unknown mode: {self.mode}")
            result = False

        if result:
            self.metrics.success_ops += 1
        else:
            self.metrics.failed_ops += 1

        return result

    async def _health_check_loop(self, interval: float = 5.0):
        """
        Enhanced health check with diagnostics for timing vs. real bugs.

        Differentiates between:
        - Transient mismatches (timing/race conditions) - EXPECTED
        - Persistent mismatches (real bugs) - ISSUE
        - Lease expirations - NEEDS ATTENTION
        - Over-capacity - CRITICAL BUG
        """
        while self.running:
            await asyncio.sleep(interval)

            try:
                for slot in self.slot_types:
                    cap = self._sema.get_capacity(slot)

                    # === FIRST READ ===
                    count1 = self._sema.read_count(slot)
                    holders1 = self._sema.list_holders(slot)
                    holder_count1 = len(holders1)

                    # Check for over-capacity (CRITICAL BUG)
                    if count1 > cap:
                        self.metrics.over_capacity_events += 1
                        logger.error(
                            f"[HEALTH] 🔴 CRITICAL: Over-capacity detected for {slot}! "
                            f"count={count1} > capacity={cap}"
                        )

                    # Check for initial mismatch
                    if count1 != holder_count1:
                        diff1 = abs(count1 - holder_count1)

                        # Wait 100ms and read again to check if it's transient
                        await asyncio.sleep(0.1)

                        # === SECOND READ ===
                        count2 = self._sema.read_count(slot)
                        holders2 = self._sema.list_holders(slot)
                        holder_count2 = len(holders2)

                        # Analyze the mismatch pattern
                        if count2 == holder_count2:
                            # Mismatch resolved itself - likely timing/race condition
                            self.metrics.counter_mismatches_transient += 1
                            logger.info(
                                f"[HEALTH] ✓ Transient mismatch for {slot} self-resolved: "
                                f"was count={count1}, holders={holder_count1} → "
                                f"now count={count2}, holders={holder_count2} (TIMING ISSUE - OK)"
                            )
                        else:
                            # Mismatch persists - more concerning
                            diff2 = abs(count2 - holder_count2)
                            self.metrics.counter_mismatches += 1
                            self.metrics.counter_mismatches_persistent += 1

                            # Detailed diagnosis
                            diagnosis = []

                            # Check if count decreased but holders increased (or vice versa)
                            if count1 > holder_count1 and count2 < holder_count2:
                                diagnosis.append("OSCILLATING")
                            elif count1 < holder_count1 and count2 > holder_count2:
                                diagnosis.append("OSCILLATING")

                            # Check for lease expiration pattern
                            if count2 > holder_count2:
                                # More count than holders - possible lease expiration
                                diagnosis.append("POSSIBLE_LEASE_EXPIRATION")

                                # Check holders for old timestamps
                                now = time.time()
                                old_holders = []
                                for tid, holder in holders2.items():
                                    try:
                                        # Parse created_at timestamp
                                        created = holder.created_at
                                        renewed = holder.renewed_at or created
                                        # This is a simplified check
                                        if (
                                            "renewed_at" not in str(holder)
                                            or holder.renewed_at is None
                                        ):
                                            old_holders.append(tid)
                                    except Exception:
                                        pass

                                if old_holders:
                                    diagnosis.append(
                                        f"STALE_HOLDERS({len(old_holders)})"
                                    )
                                    self.metrics.lease_expirations_detected += 1

                            diagnosis_str = (
                                ", ".join(diagnosis) if diagnosis else "UNKNOWN"
                            )

                            logger.warning(
                                f"[HEALTH] ⚠️  PERSISTENT mismatch for {slot}: "
                                f"1st read: count={count1}, holders={holder_count1} (diff={diff1}) → "
                                f"2nd read: count={count2}, holders={holder_count2} (diff={diff2}) "
                                f"[{diagnosis_str}]"
                            )

                            # Auto-reconcile persistent mismatches
                            new_count = self._sema.reconcile(slot)
                            self.metrics.reconciliations += 1
                            logger.info(
                                f"[HEALTH] Reconciled {slot}: {count2} → {new_count}"
                            )

                            # Verify reconciliation worked
                            await asyncio.sleep(0.05)
                            verify_count = self._sema.read_count(slot)
                            verify_holders = len(self._sema.list_holders(slot))
                            if verify_count != verify_holders:
                                logger.error(
                                    f"[HEALTH] 🔴 RECONCILIATION FAILED for {slot}! "
                                    f"Still mismatched: count={verify_count}, holders={verify_holders}"
                                )

            except Exception as e:
                logger.error(f"[HEALTH] Check failed: {e}", exc_info=True)

    async def run(
        self,
        num_requests: int = 0,
        run_time: str = "30s",
        concurrency: int = 30,
    ):
        """Run the stress test"""
        duration = self._parse_duration(run_time)
        end_time = (
            (time.monotonic() + duration.total_seconds())
            if duration.total_seconds() > 0
            else None
        )

        if num_requests <= 0 and not end_time:
            raise ValueError("Either num_requests (>0) or run_time must be specified")

        logger.info(
            f"Starting State Management stress test:\n"
            f"  mode: {self.mode}\n"
            f"  duration: {run_time if not num_requests else f'{num_requests} ops'}\n"
            f"  concurrency: {concurrency}\n"
            f"  slots: {self.slot_types}\n"
            f"  workers: {self.num_workers}"
        )

        # Initialize capacities
        self._init_capacities()

        # Initial reconciliation
        for slot in self.slot_types:
            try:
                new_cnt = self._sema.reconcile(slot)
                logger.info(f"Initial reconcile for {slot}: count={new_cnt}")
            except Exception as e:
                logger.warning(f"Initial reconcile failed for {slot}: {e}")

        # Start health check loop
        health_task = asyncio.create_task(self._health_check_loop())

        started = time.time()
        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = set()
            idx = 0

            # Distribute operations across workers
            worker_idx = 0

            while self.running:
                if end_time and time.monotonic() >= end_time:
                    break
                if num_requests > 0 and idx >= num_requests:
                    break

                while len(futures) < concurrency and (
                    num_requests == 0 or idx < num_requests
                ):
                    if end_time and time.monotonic() >= end_time:
                        break

                    worker_id = f"w{worker_idx % self.num_workers}"
                    fut = loop.run_in_executor(
                        executor, self._perform_one, worker_id, idx
                    )
                    futures.add(fut)
                    idx += 1
                    worker_idx += 1

                if not futures:
                    break

                done, futures = await asyncio.wait(
                    futures, return_when=asyncio.FIRST_COMPLETED
                )

                for f in done:
                    try:
                        f.result()
                    except Exception as e:
                        logger.error(f"Operation error: {e}")

        # Stop health check
        self.running = False
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

        elapsed = time.time() - started
        self._print_summary(elapsed)

    def _print_summary(self, elapsed: float):
        """Print comprehensive summary"""
        tps = self.metrics.total_ops / elapsed if elapsed > 0 else 0.0

        logger.info("\n" + "=" * 70)
        logger.info("STATE MANAGEMENT STRESS TEST SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\n[CONFIG]")
        logger.info(f"  Mode: {self.mode}")
        logger.info(
            f"  Slots: {len(self.slot_types)} x {self.capacity_per_slot} capacity"
        )
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Duration: {elapsed:.2f}s")

        logger.info(f"\n[OPERATIONS]")
        logger.info(f"  Total:   {self.metrics.total_ops:,} ops ({tps:,.2f} ops/s)")
        logger.info(
            f"  Success: {self.metrics.success_ops:,} ({100*self.metrics.success_ops/max(1,self.metrics.total_ops):.1f}%)"
        )
        logger.info(
            f"  Failed:  {self.metrics.failed_ops:,} ({100*self.metrics.failed_ops/max(1,self.metrics.total_ops):.1f}%)"
        )

        logger.info(f"\n[BREAKDOWN]")
        logger.info(
            f"  Reserves: {self.metrics.reserve_success:,} ✓  {self.metrics.reserve_failed:,} ✗"
        )
        logger.info(
            f"  Releases: {self.metrics.release_success:,} ✓  {self.metrics.release_failed:,} ✗"
        )
        logger.info(
            f"  Renewals: {self.metrics.renew_success:,} ✓  {self.metrics.renew_failed:,} ✗"
        )

        logger.info(f"\n[HEALTH]")
        logger.info(f"  Ownership violations:      {self.metrics.ownership_violations}")
        logger.info(f"  Counter mismatches:")
        logger.info(
            f"    - Transient (timing):    {self.metrics.counter_mismatches_transient} ✓ (self-resolved)"
        )
        logger.info(
            f"    - Persistent (real):     {self.metrics.counter_mismatches_persistent} ⚠️  (needed reconcile)"
        )
        logger.info(
            f"    - Total:                 {self.metrics.counter_mismatches_transient + self.metrics.counter_mismatches_persistent}"
        )
        logger.info(f"  Reconciliations:           {self.metrics.reconciliations}")
        logger.info(
            f"  Lease expirations detected: {self.metrics.lease_expirations_detected}"
        )
        logger.info(f"  Over-capacity events:      {self.metrics.over_capacity_events}")
        logger.info(f"  Capacity changes:          {self.metrics.capacity_changes}")

        # Per-slot status
        logger.info(f"\n[SLOT STATUS]")
        over_capacity_detected = False
        for slot in self.slot_types:
            try:
                cap = self._sema.get_capacity(slot)
                used = self._sema.read_count(slot)
                avail = self._sema.available_slot_count(slot)
                holders = len(self._sema.list_holders(slot))

                issues = []
                if used != holders:
                    issues.append("COUNT≠HOLDERS")
                if used > cap:
                    issues.append("OVER_CAPACITY")
                    over_capacity_detected = True

                status = "⚠️ " + ", ".join(issues) if issues else "✓"

                logger.info(
                    f"  {slot:12s}: cap={cap:2d}, used={used:2d}, avail={avail:2d}, "
                    f"holders={holders:2d} {status}"
                )
            except Exception as e:
                logger.error(f"  {slot:12s}: Error reading status - {e}")

        # Per-worker stats
        if self.worker_metrics:
            logger.info(f"\n[TOP WORKERS]")
            top_workers = sorted(
                self.worker_metrics.values(),
                key=lambda w: w.reserves_success,
                reverse=True,
            )[:5]
            for wm in top_workers:
                logger.info(
                    f"  {wm.worker_id}: "
                    f"reserves={wm.reserves_success:,}, "
                    f"releases={wm.releases_success:,}, "
                    f"renewals={wm.renewals_success:,}, "
                    f"violations={wm.ownership_violations}"
                )

        # Final verdict
        logger.info(f"\n[VERDICT]")

        # Critical issues
        if over_capacity_detected or self.metrics.over_capacity_events > 0:
            logger.error(
                "  🔴 CRITICAL: OVER-CAPACITY DETECTED - Semaphore allowed more reserves than capacity!"
            )

        if self.metrics.ownership_violations > 0:
            logger.error("  ❌ SECURITY VIOLATIONS DETECTED!")

        # Mismatch analysis
        total_mismatches = (
            self.metrics.counter_mismatches_transient
            + self.metrics.counter_mismatches_persistent
        )
        if total_mismatches > 0:
            transient_pct = (
                100 * self.metrics.counter_mismatches_transient / total_mismatches
            )
            if transient_pct >= 80:
                logger.info(
                    f"  ✓ Counter mismatches: {transient_pct:.0f}% transient (timing issues) - EXPECTED under high load"
                )
            elif self.metrics.counter_mismatches_persistent > 10:
                logger.warning(
                    f"  ⚠️  High persistent mismatches ({self.metrics.counter_mismatches_persistent}) - May indicate issue"
                )

        if self.metrics.lease_expirations_detected > 0:
            logger.warning(
                f"  ⚠️  Lease expirations detected ({self.metrics.lease_expirations_detected}) - "
                "Consider more frequent renewal or longer TTL"
            )

        success_rate = 100 * self.metrics.success_ops / max(1, self.metrics.total_ops)
        if success_rate >= 95:
            logger.info(f"  ✅ SUCCESS RATE: {success_rate:.1f}%")
        elif success_rate >= 50:
            logger.warning(f"  ⚠️  SUCCESS RATE: {success_rate:.1f}%")
        else:
            logger.warning(
                f"  ⚠️  LOW SUCCESS RATE: {success_rate:.1f}% (may indicate capacity exhaustion)"
            )

        logger.info("=" * 70 + "\n")

    def stop(self):
        """Stop the stress test"""
        self.running = False
        try:
            self._client.close()
        except Exception:
            pass


# ============================= CLI =============================


async def _amain(args):
    stresser = StateManagementStresser(
        etcd_host=args.host,
        etcd_port=args.port,
        mode=args.mode,
        num_slots=args.num_slots,
        num_workers=args.num_workers,
        capacity_per_slot=args.capacity,
        lease_ttl=args.lease_ttl,
        enable_chaos=args.chaos,
    )
    try:
        await stresser.run(
            num_requests=args.num_requests,
            run_time=args.run_time,
            concurrency=args.concurrency,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stresser.stop()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comprehensive state management stress tester"
    )
    p.add_argument("--host", type=str, default="localhost", help="etcd host")
    p.add_argument("--port", type=int, default=2379, help="etcd port")
    p.add_argument(
        "--mode",
        type=str,
        choices=[
            "reserve_release",
            "reserve_release_owned",
            "renew_stress",
            "capacity_changes",
            "owner_validation",
            "counter_recovery",
            "slot_manager",
            "full_integration",
        ],
        default="full_integration",
        help="stress test mode",
    )
    p.add_argument("--num-slots", type=int, default=3, help="number of slot types")
    p.add_argument(
        "--num-workers", type=int, default=10, help="number of simulated workers"
    )
    p.add_argument("--capacity", type=int, default=20, help="capacity per slot type")
    p.add_argument(
        "--lease-ttl", type=int, default=30, help="holder lease TTL (seconds)"
    )
    p.add_argument(
        "--chaos", action="store_true", help="enable chaos mode (delete counters, etc.)"
    )
    p.add_argument(
        "--num-requests", type=int, default=0, help="total ops (0 uses duration)"
    )
    p.add_argument(
        "--run-time", type=str, default="30s", help="duration (e.g., 30s, 2m, 1h)"
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="concurrent operations (recommend <50% of total capacity)",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    # Example usage:
    #
    # Basic reserve/release:
    #   python -m tools.stress.state_management_stresser --mode reserve_release --run-time 30s
    #
    # Test ownership validation:
    #   python -m tools.stress.state_management_stresser --mode owner_validation --num-requests 1000
    #
    # Lease renewal stress:
    #   python -m tools.stress.state_management_stresser --mode renew_stress --lease-ttl 5 --run-time 60s
    #   python -m tools.stress.state_management_stresser --mode renew_stress --lease-ttl 10 --run-time 20s --num-workers 5 --num-slots 2 --capacity 3
    #
    # Dynamic capacity changes:
    #   python -m tools.stress.state_management_stresser --mode capacity_changes --run-time 45s
    #
    # Counter recovery (Bug #2 test):
    #   python -m tools.stress.state_management_stresser --mode counter_recovery --chaos --run-time 30s
    #   python -m tools.stress.state_management_stresser --mode counter_recovery --run-time 30s --num-workers 5 --concurrency 10 --capacity 10
    #
    # SlotCapacityManager integration:
    #   python -m tools.stress.state_management_stresser --mode slot_manager --num-slots 5 --run-time 60s
    #
    # Full integration (recommended):
    #   python -m tools.stress.state_management_stresser --mode full_integration --num-slots 5 --num-workers 20 --capacity 10 --concurrency 100 --run-time 2m
    #
    main()
