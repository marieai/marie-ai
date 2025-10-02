import argparse
import asyncio
import logging
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Literal, Optional

from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.semaphore_store import SemaphoreStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SemaphoreStoreStresser")

Mode = Literal[
    "set_capacity",  # set capacity only
    "reserve",  # reserve only (requires capacity)
    "release",  # reserve -> release
    "reconcile",  # reconcile only
    "end_to_end",  # set capacity -> reserve -> release (loop)
]


class SemaphoreStoreStresser:
    """
    Stress tester for SemaphoreStore flows:
      - set_capacity: set capacity for a given slot_type repeatedly
      - reserve: attempts atomic reservations (requires capacity set)
      - release: reserves then releases (validates reserve/release CAS flow)
      - reconcile: recomputes count from holders and CAS-updates counter
      - end_to_end: ensures capacity, then reserve + release in a loop

    By default, a unique ticket is used per operation. You can fix the slot_type for concentrated contention testing.
    """

    def __init__(
        self,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        mode: Mode = "end_to_end",
        slot_type: str = "default",
        capacity: int = 10,
        lease_ttl: int = 30,
    ):
        self._client = EtcdClient(etcd_host=etcd_host, etcd_port=etcd_port)
        self._sema = SemaphoreStore(self._client, default_lease_ttl=lease_ttl)
        self.mode: Mode = mode
        self.slot_type = slot_type
        self.capacity = int(capacity)
        self.running = True

        logger.info(
            f"SemaphoreStoreStresser initialized at {etcd_host}:{etcd_port} "
            f"mode={self.mode} slot_type={self.slot_type} capacity={self.capacity} ttl={lease_ttl}"
        )

    # ------------- Helpers -------------

    @staticmethod
    def _parse_duration(duration_str: str) -> timedelta:
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

    def _ensure_capacity(self) -> None:
        cur = self._sema.get_capacity(self.slot_type)
        if cur is None or cur != self.capacity:
            self._sema.set_capacity(self.slot_type, self.capacity)

    # ------------- Single ops -------------

    def _op_set_capacity(self, idx: int) -> bool:
        try:
            # vary capacity a bit to stress CAS around reads in reserve
            cap = max(1, self.capacity + random.randint(-2, 2))
            self._sema.set_capacity(self.slot_type, cap)
            return True
        except Exception as e:
            logger.error(f"set_capacity error: {e}", exc_info=False)
            return False

    def _op_reserve(self, idx: int) -> bool:
        try:
            self._ensure_capacity()
            # Lightweight retry to overcome CAS contention (not capacity exhaustion)
            max_attempts = 5
            base_sleep = 0.001  # 1 ms
            for attempt in range(1, max_attempts + 1):
                ticket_id = f"t-{uuid.uuid4()}"
                ok = self._sema.reserve(
                    slot_type=self.slot_type,
                    ticket_id=ticket_id,
                    job_id=f"job-{idx}",
                    node=f"node-{idx % 32}",
                    ttl=None,
                )
                if ok:
                    return True

                # If we appear starved, don't spin aggressively
                try:
                    avail = self._sema.available_slot_count(self.slot_type)
                except Exception:
                    avail = None

                # Back off a bit (jitter)
                if avail is None or avail > 0:
                    time.sleep(base_sleep * attempt * (1.0 + random.random()))
                else:
                    # No capacity reported; bail fast
                    break

            return False
        except Exception as e:
            logger.error(f"reserve error: {e}", exc_info=False)
            return False

    def _op_release(self, idx: int) -> bool:
        try:
            self._ensure_capacity()
            ticket_id = f"t-{uuid.uuid4()}"
            ok_r = self._sema.reserve(
                slot_type=self.slot_type,
                ticket_id=ticket_id,
                job_id=f"job-{idx}",
                node=f"node-{idx % 32}",
                ttl=None,
            )
            if not ok_r:
                return False
            ok_rel = self._sema.release(self.slot_type, ticket_id)
            return bool(ok_rel)
        except Exception as e:
            logger.error(f"release error: {e}", exc_info=False)
            return False

    def _op_reconcile(self, idx: int) -> bool:
        try:
            # reconciliation is safe regardless of capacity but ensure it's set
            self._ensure_capacity()
            _ = self._sema.reconcile(self.slot_type)
            return True
        except Exception as e:
            logger.error(f"reconcile error: {e}", exc_info=False)
            return False

    def _op_end_to_end(self, idx: int) -> bool:
        try:
            self._ensure_capacity()
            # retry reserve a few times to handle CAS contention
            max_attempts = 5
            base_sleep = 0.001
            for attempt in range(1, max_attempts + 1):
                ticket_id = f"t-{uuid.uuid4()}"
                ok_r = self._sema.reserve(
                    slot_type=self.slot_type,
                    ticket_id=ticket_id,
                    job_id=f"job-{idx}",
                    node=f"node-{idx % 32}",
                    ttl=None,
                )
                if ok_r:
                    # probabilistically release
                    if random.random() < 0.8:
                        ok_rel = self._sema.release(self.slot_type, ticket_id)
                        return bool(ok_rel)
                    else:
                        return True

                try:
                    avail = self._sema.available_slot_count(self.slot_type)
                except Exception:
                    avail = None

                if avail is None or avail > 0:
                    time.sleep(base_sleep * attempt * (1.0 + random.random()))
                else:
                    break

            return False
        except Exception as e:
            logger.error(f"end_to_end error: {e}", exc_info=False)
            return False

    def _perform_one(self, idx: int) -> bool:
        if self.mode == "set_capacity":
            return self._op_set_capacity(idx)
        if self.mode == "reserve":
            return self._op_reserve(idx)
        if self.mode == "release":
            return self._op_release(idx)
        if self.mode == "reconcile":
            return self._op_reconcile(idx)
        if self.mode == "end_to_end":
            return self._op_end_to_end(idx)
        logger.error(f"Unknown mode: {self.mode}")
        return False

    async def run(
        self,
        num_requests: int = 0,
        run_time: str = "30s",
        concurrency: int = 50,
        max_in_flight: Optional[int] = None,
    ):
        duration = self._parse_duration(run_time)
        end_time = (
            (time.monotonic() + duration.total_seconds())
            if duration.total_seconds() > 0
            else None
        )
        cap = max_in_flight or concurrency

        if num_requests <= 0 and not end_time:
            raise ValueError("Either num_requests (>0) or run_time must be specified")

        logger.info(
            f"Starting Semaphore stress: mode={self.mode}, slot_type={self.slot_type}, "
            f"{'num_requests=' + str(num_requests) if num_requests else 'run_time=' + run_time}, "
            f"concurrency={concurrency}, cap={cap}"
        )

        # Ensure capacity and reconcile once before starting, so we don't start from a stale counter.
        try:
            self._ensure_capacity()
            try:
                new_cnt = self._sema.reconcile(self.slot_type)
                logger.info(
                    f"Pre-run reconcile for slot_type={self.slot_type} set count={new_cnt}"
                )
            except Exception as e:
                logger.warning(f"Pre-run reconcile skipped: {e}")
        except Exception as e:
            logger.error(f"Failed to ensure capacity before run: {e}")

        success = 0
        failure = 0
        started = time.time()

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = set()
            idx = 0

            # Periodic reconcile when starved (no available slots for some time)
            last_reconcile = 0.0
            reconcile_interval_s = 5.0

            while self.running:
                if end_time and time.monotonic() >= end_time:
                    break
                if num_requests > 0 and idx >= num_requests:
                    break

                # Optional: if we appear starved for a while, try a reconcile to correct counter drift
                try:
                    avail = self._sema.available_slot_count(self.slot_type)
                    if (
                        avail <= 0
                        and (time.time() - last_reconcile) >= reconcile_interval_s
                    ):
                        new_cnt = self._sema.reconcile(self.slot_type)
                        last_reconcile = time.time()
                        logger.info(
                            f"Auto-reconcile triggered due to starvation (avail={avail}). "
                            f"slot_type={self.slot_type} count_now={new_cnt}"
                        )
                except Exception as e:
                    logger.warning(f"Auto-reconcile check failed: {e}")

                while len(futures) < cap and (num_requests == 0 or idx < num_requests):
                    if end_time and time.monotonic() >= end_time:
                        break
                    fut = loop.run_in_executor(executor, self._perform_one, idx)
                    futures.add(fut)
                    idx += 1

                if not futures:
                    break

                done, futures = await asyncio.wait(
                    futures, return_when=asyncio.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        if f.result():
                            success += 1
                        else:
                            failure += 1
                    except Exception as e:
                        logger.error(f"in-flight op error: {e}", exc_info=False)
                        failure += 1

        elapsed = time.time() - started
        total = success + failure
        tps = total / elapsed if elapsed > 0 else 0.0

        # quick introspection
        cur_cap = self._sema.get_capacity(self.slot_type)
        used = self._sema.read_count(self.slot_type)
        avail = self._sema.available_slot_count(self.slot_type)

        logger.info("---- Semaphore Stress Summary ----")
        logger.info(f"Mode: {self.mode}, SlotType: {self.slot_type}")
        logger.info(
            f"Ops: {total:,} (success={success:,}, failure={failure:,}) in {elapsed:.2f}s"
        )
        logger.info(f"Throughput: {tps:,.2f} ops/s")
        logger.info(f"Capacity={cur_cap}, Used={used}, Available={avail}")
        logger.info("---------------------------------")

    def stop(self):
        self.running = False
        try:
            self._client.close()
        except Exception:
            pass


# ----------------------------- CLI -----------------------------


async def _amain(args):
    stresser = SemaphoreStoreStresser(
        etcd_host=args.host,
        etcd_port=args.port,
        mode=args.mode,
        slot_type=args.slot_type,
        capacity=args.capacity,
        lease_ttl=args.lease_ttl,
    )
    try:
        await stresser.run(
            num_requests=args.num_requests,
            run_time=args.run_time,
            concurrency=args.concurrency,
            max_in_flight=args.max_in_flight,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stresser.stop()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SemaphoreStore stress tester")
    p.add_argument("--host", type=str, default="localhost", help="etcd host")
    p.add_argument("--port", type=int, default=2379, help="etcd port")
    p.add_argument(
        "--mode",
        type=str,
        choices=["set_capacity", "reserve", "release", "reconcile", "end_to_end"],
        default="end_to_end",
        help="stress mode",
    )
    p.add_argument("--slot-type", type=str, default="default", help="slot type name")
    p.add_argument("--capacity", type=int, default=10, help="capacity to set/ensure")
    p.add_argument(
        "--lease-ttl", type=int, default=30, help="default holder lease ttl (seconds)"
    )
    p.add_argument(
        "--num-requests", type=int, default=0, help="total ops (0 uses duration)"
    )
    p.add_argument(
        "--run-time", type=str, default="30s", help="duration (e.g., 30s, 2m, 1h)"
    )
    p.add_argument("--concurrency", type=int, default=50, help="number of workers")
    p.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="cap on in-flight futures (defaults to concurrency)",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    # Examples:
    # python -m tools.stress.semaphore_store_stresser --mode set_capacity --slot-type gpu --capacity 8
    # python -m tools.stress.semaphore_store_stresser --mode reserve --run-time 30s --concurrency 64
    # python -m tools.stress.semaphore_store_stresser --mode release --num-requests 50000 --concurrency 32
    # python -m tools.stress.semaphore_store_stresser --mode reconcile --slot-type gpu --run-time 15s
    # python -m tools.stress.semaphore_store_stresser --mode end_to_end --slot-type cpu --capacity 16 --run-time 45s
    main()
