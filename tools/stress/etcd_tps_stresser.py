import argparse
import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Literal, Optional

from marie.serve.discovery.etcd_client import EtcdClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


Mode = Literal["put", "cas_value", "cas_mod", "txn"]


class EtcdTpsStresser:
    """
    A stress tester for EtcdClient to measure throughput in different modes:
      - put: plain single PUT per operation
      - cas_value: CAS based on value equality (compare_and_set)
      - cas_mod: CAS based on mod_revision (update_if_unchanged)
      - txn: simple transactional if_missing().put().commit()

    Each operation is independent and uses a unique key to avoid key conflicts unless otherwise configured.
    """

    def __init__(
        self, etcd_host: str = "localhost", etcd_port: int = 2379, mode: Mode = "put"
    ):
        """
        :param etcd_host: The hostname or IP of the etcd server.
        :param etcd_port: The port of the etcd server.
        :param mode: Operation mode: put | cas_value | cas_mod | txn
        """
        self.etcd_client = EtcdClient(etcd_host=etcd_host, etcd_port=etcd_port)
        self.running = True
        self.mode: Mode = mode
        logger.info(
            f"Stresser initialized for etcd at {etcd_host}:{etcd_port}, mode={self.mode}"
        )

    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parses a duration string like '10h', '30m', '5s' into a timedelta."""
        if not duration_str:
            return timedelta()
        match = re.match(r"(\d+)\s*([hms])?$", duration_str.lower().strip())
        if not match:
            raise ValueError("Invalid duration format. Use '10h', '30m', or '5s'.")

        value = int(match.group(1))
        unit = match.group(2) or "s"

        if unit == "h":
            return timedelta(hours=value)
        elif unit == "m":
            return timedelta(minutes=value)
        else:  # 's'
            return timedelta(seconds=value)

    # --------- Single operation runners ---------

    def _op_put(self, idx: int) -> bool:
        """
        Single PUT operation with a unique key.
        """
        key = f"stress/put/{uuid.uuid4()}"
        val = f"payload-{idx}"
        self.etcd_client.put(key, val)
        return True

    def _op_cas_value(self, idx: int) -> bool:
        """
        CAS using value equality (compare_and_set):
          1) Initialize key to 'v1'
          2) CAS 'v1' -> 'v2'
        """
        key = f"stress/cas_value/{uuid.uuid4()}"
        try:
            self.etcd_client.put(key, "v1")
            return bool(self.etcd_client.compare_and_set(key, "v1", "v2"))
        except Exception as e:
            logger.error(f"cas_value error: {e}", exc_info=False)
            return False

    def _op_cas_mod(self, idx: int) -> bool:
        """
        CAS using mod_revision (update_if_unchanged):
          1) Initialize key to 'v0'
          2) Read meta.mod_revision
          3) Attempt update_if_unchanged with that revision
        """
        key = f"stress/cas_mod/{uuid.uuid4()}"
        try:
            self.etcd_client.put(key, "v0")
            _, meta = self.etcd_client.get(key, metadata=True, serializable=False)
            if not meta or not hasattr(meta, "mod_revision"):
                return False
            return bool(
                self.etcd_client.update_if_unchanged(key, "v1", meta.mod_revision)
            )
        except Exception as e:
            logger.error(f"cas_mod error: {e}", exc_info=False)
            return False

    def _op_txn(self, idx: int) -> bool:
        """
        Simple transaction using fluent API:
          if_missing(key).put(key, value).commit()
        """
        key = f"stress/txn/{uuid.uuid4()}"
        try:
            with self.etcd_client.txn() as t:
                t.if_missing(key).put(key, f"txn-{idx}")
                ok, _ = t.commit()
                return bool(ok)
        except Exception as e:
            logger.error(f"txn error: {e}", exc_info=False)
            return False

    def _perform_one(self, request_idx: int) -> bool:
        """
        Dispatch a single operation based on the configured mode.
        """
        try:
            if self.mode == "put":
                return self._op_put(request_idx)
            elif self.mode == "cas_value":
                return self._op_cas_value(request_idx)
            elif self.mode == "cas_mod":
                return self._op_cas_mod(request_idx)
            elif self.mode == "txn":
                return self._op_txn(request_idx)
            else:
                logger.error(f"Unknown mode: {self.mode}")
                return False
        except Exception as e:
            logger.error(f"Operation error in mode={self.mode}: {e}", exc_info=False)
            return False

    # --------- Main entry ---------

    async def run_stress_test(
        self,
        num_requests: int = 0,
        run_time: str = "",
        concurrency: int = 50,
        max_in_flight: Optional[int] = None,
    ):
        """
        Runs a high-volume stress test for a specified number of requests or a duration.

        :param num_requests: Total number of requests to simulate. Stops when this is reached (0 means unlimited if run_time given).
        :param run_time: Duration to run the test (e.g., "30s", "2m"). Stops after this time.
        :param concurrency: Number of parallel threads sending requests.
        :param max_in_flight: Optional cap on futures in flight (defaults to 'concurrency').
        """
        duration = self._parse_duration(run_time)
        end_time = (
            (time.monotonic() + duration.total_seconds())
            if duration.total_seconds() > 0
            else None
        )
        cap = max_in_flight or concurrency

        if num_requests <= 0 and not end_time:
            raise ValueError(
                "Either `num_requests` (> 0) or `run_time` must be specified."
            )

        logger.info(
            f"Starting stress test: mode={self.mode}, "
            f"{'run_time='+run_time if end_time else ''} "
            f"{'num_requests='+str(num_requests) if num_requests > 0 else ''} "
            f"concurrency={concurrency}"
        )

        success_count = 0
        failure_count = 0
        start_time_run = time.time()

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = set()
            request_index = 0

            while self.running:
                # stopping conditions
                if end_time and time.monotonic() >= end_time:
                    logger.info("Run time limit reached. Finishing up...")
                    break
                if num_requests > 0 and request_index >= num_requests:
                    logger.info("Request limit reached. Finishing up...")
                    break

                # fill the pipeline
                while len(futures) < cap and (
                    num_requests <= 0 or request_index < num_requests
                ):
                    if end_time and time.monotonic() >= end_time:
                        break
                    fut = loop.run_in_executor(
                        executor, self._perform_one, request_index
                    )
                    futures.add(fut)
                    request_index += 1

                if not futures:
                    break

                done, futures = await asyncio.wait(
                    futures, return_when=asyncio.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        if f.result():
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        logger.error(f"in-flight future error: {e}", exc_info=False)
                        failure_count += 1

        total_duration = time.time() - start_time_run
        total_processed = success_count + failure_count
        tps = total_processed / total_duration if total_duration > 0 else 0

        logger.info("--- Stress Test Summary ---")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Total operations attempted: {total_processed:,}")
        logger.info(f"Successful: {success_count:,}, Failed: {failure_count:,}")
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Throughput: {tps:,.2f} ops/sec")
        logger.info("--------------------------")

    def stop(self):
        """Stops the stress test gracefully."""
        logger.info("Stopping stress test...")
        self.running = False
        try:
            self.etcd_client.close()
        except Exception:
            pass


async def _amain(args):
    stresser = EtcdTpsStresser(etcd_host=args.host, etcd_port=args.port, mode=args.mode)
    try:
        await stresser.run_stress_test(
            num_requests=args.num_requests,
            run_time=args.run_time,
            concurrency=args.concurrency,
            max_in_flight=args.max_in_flight,
        )
    except KeyboardInterrupt:
        logger.info("Stress test interrupted by user.")
    finally:
        stresser.stop()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Etcd TPS/CAS/Txn stresser")
    p.add_argument("--host", type=str, default="localhost", help="etcd host")
    p.add_argument("--port", type=int, default=2379, help="etcd port")
    p.add_argument(
        "--mode",
        type=str,
        choices=["put", "cas_value", "cas_mod", "txn"],
        default="put",
        help="Operation mode",
    )
    p.add_argument(
        "--num-requests",
        type=int,
        default=0,
        help="Total operations to perform (0 means unlimited if run_time provided)",
    )
    p.add_argument(
        "--run-time", type=str, default="30s", help="Run duration, e.g., 30s, 2m, 1h"
    )
    p.add_argument("--concurrency", type=int, default=50, help="Parallel workers")
    p.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Optional cap on in-flight operations (defaults to concurrency)",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    # Examples:
    #   python -m tools.stress.etcd_tps_stresser --mode put --run-time 30s --concurrency 25
    #   python -m tools.stress.etcd_tps_stresser --mode cas_value --num-requests 10000 --concurrency 50
    #   python -m tools.stress.etcd_tps_stresser --mode cas_mod --run-time 60s --concurrency 32
    #   python -m tools.stress.etcd_tps_stresser --mode txn --run-time 45s --concurrency 16
    main()
