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

from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.state_store import DesiredStore, StatusStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DesiredStatusStresser")


Mode = Literal[
    "desired_set",  # DesiredStore.set
    "desired_schedule",  # DesiredStore.schedule_new_epoch
    "status_claim",  # StatusStore.claim
    "status_set",  # StatusStore.set_serving / set_not_serving / set_unknown / set_service_unknown
    "heartbeat",  # StatusStore.heartbeat
    "end_to_end",  # desired_schedule -> status_claim -> set_serving -> heartbeat
]


class DesiredStatusStresser:
    """
    Stress tester focusing on DesiredStore and StatusStore flows:
      - desired_set: creates/updates desired doc with set(...)
      - desired_schedule: schedule_new_epoch(...) repeatedly
      - status_claim: claim(...) ownership of /status
      - status_set: flip status through common states for owned entry
      - heartbeat: call heartbeat(...) on owned entry
      - end_to_end: schedule desired -> claim -> set SERVING -> heartbeat
    Each iteration uses (node, depl) produced per-op unless --fixed-node/--fixed-depl are set.
    """

    def __init__(
        self,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        mode: Mode = "end_to_end",
        fixed_node: Optional[str] = None,
        fixed_depl: Optional[str] = None,
    ):
        self._client = EtcdClient(etcd_host=etcd_host, etcd_port=etcd_port)
        self._desired = DesiredStore(self._client)
        self._status = StatusStore(self._client)
        self.mode: Mode = mode
        self.fixed_node = fixed_node
        self.fixed_depl = fixed_depl
        self.running = True

        logger.info(
            f"DesiredStatusStresser initialized at {etcd_host}:{etcd_port} "
            f"mode={self.mode} fixed_node={self.fixed_node} fixed_depl={self.fixed_depl}"
        )

    # ------------- Helpers -------------

    @staticmethod
    def _parse_duration(duration_str: str) -> timedelta:
        if not duration_str:
            return timedelta()
        m = re.match(r"(\d+)\s*([hms])?$", duration_str.lower().strip())
        if not m:
            raise ValueError("Invalid duration. Use forms like '30s', '2m', or '1h'")
        value = int(m.group(1))
        unit = m.group(2) or "s"
        if unit == "h":
            return timedelta(hours=value)
        elif unit == "m":
            return timedelta(minutes=value)
        return timedelta(seconds=value)

    def _ids(self) -> tuple[str, str]:
        node = self.fixed_node or f"node-{uuid.uuid4()}"
        depl = self.fixed_depl or f"depl-{uuid.uuid4()}"
        return node, depl

    # ------------- Single ops -------------

    def _op_desired_set(self, idx: int) -> bool:
        node, depl = self._ids()
        try:
            params = {"idx": idx, "rand": random.randint(0, 999)}
            _ = self._desired.set(node, depl, params=params, phase="SCHEDULED")
            return True
        except Exception as e:
            logger.error(f"desired_set error: {e}", exc_info=False)
            return False

    def _op_desired_schedule(self, idx: int) -> bool:
        node, depl = self._ids()
        try:
            params = {"job": f"job-{idx}"}
            _ = self._desired.schedule_new_epoch(node, depl, params=params)
            return True
        except Exception as e:
            logger.error(f"desired_schedule error: {e}", exc_info=False)
            return False

    def _op_status_claim(self, idx: int) -> bool:
        node, depl = self._ids()
        try:
            owner = f"worker-{uuid.uuid4()}"
            epoch = random.randint(1, 3)
            return bool(
                self._status.claim(
                    node,
                    depl,
                    worker_id=owner,
                    epoch=epoch,
                    initial_status=HealthCheckResponse.NOT_SERVING,
                )
            )
        except Exception as e:
            logger.error(f"status_claim error: {e}", exc_info=False)
            return False

    def _op_status_set(self, idx: int) -> bool:
        node, depl = self._ids()
        try:
            owner = f"worker-{uuid.uuid4()}"
            # Ensure entry exists and owner matches
            self._status.claim(
                node,
                depl,
                owner,
                epoch=1,
                initial_status=HealthCheckResponse.NOT_SERVING,
            )

            # Flip through a small cycle
            ok1 = self._status.set_serving(node, depl, owner)
            ok2 = self._status.set_not_serving(node, depl, owner)
            ok3 = self._status.set_unknown(node, depl, owner)
            ok4 = self._status.set_service_unknown(node, depl, owner)
            return bool(ok1 and ok2 and ok3 and ok4)
        except Exception as e:
            logger.error(f"status_set error: {e}", exc_info=False)
            return False

    def _op_heartbeat(self, idx: int) -> bool:
        node, depl = self._ids()
        try:
            owner = f"worker-{uuid.uuid4()}"
            # claim then heartbeat
            self._status.claim(
                node,
                depl,
                owner,
                epoch=1,
                initial_status=HealthCheckResponse.NOT_SERVING,
            )
            return bool(self._status.heartbeat(node, depl, owner))
        except Exception as e:
            logger.error(f"heartbeat error: {e}", exc_info=False)
            return False

    def _op_end_to_end(self, idx: int) -> bool:
        node, depl = self._ids()
        try:
            # gateway-like desired schedule
            _ = self._desired.schedule_new_epoch(node, depl, params={"idx": idx})

            # worker claims and sets SERVING
            owner = f"worker-{uuid.uuid4()}"
            ok_claim = self._status.claim(
                node,
                depl,
                owner,
                epoch=1,
                initial_status=HealthCheckResponse.NOT_SERVING,
            )
            if not ok_claim:
                return False

            ok_serving = self._status.set_serving(node, depl, owner)
            if not ok_serving:
                return False

            # heartbeat once
            ok_hb = self._status.heartbeat(node, depl, owner)
            return bool(ok_hb)
        except Exception as e:
            logger.error(f"end_to_end error: {e}", exc_info=False)
            return False

    def _perform_one(self, idx: int) -> bool:
        if self.mode == "desired_set":
            return self._op_desired_set(idx)
        if self.mode == "desired_schedule":
            return self._op_desired_schedule(idx)
        if self.mode == "status_claim":
            return self._op_status_claim(idx)
        if self.mode == "status_set":
            return self._op_status_set(idx)
        if self.mode == "heartbeat":
            return self._op_heartbeat(idx)
        if self.mode == "end_to_end":
            return self._op_end_to_end(idx)
        logger.error(f"Unknown mode: {self.mode}")
        return False

    # ------------- Main entry -------------

    async def run(
        self,
        num_requests: int = 0,
        run_time: str = "30s",
        concurrency: int = 50,
        max_in_flight: Optional[int] = None,
    ):
        """
        Run the stress test.

        :param num_requests: max operations to execute; if 0, use duration
        :param run_time: duration like '30s', '2m' when num_requests == 0
        :param concurrency: thread pool size
        :param max_in_flight: max futures in flight (defaults to concurrency)
        """
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
            f"Starting Desired/Status stress: mode={self.mode}, "
            f"{'num_requests='+str(num_requests) if num_requests else 'run_time='+run_time}, "
            f"concurrency={concurrency}, cap={cap}"
        )

        success = 0
        failure = 0
        started = time.time()

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = set()
            idx = 0

            while self.running:
                if end_time and time.monotonic() >= end_time:
                    break
                if num_requests > 0 and idx >= num_requests:
                    break

                # fill pipeline
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

        logger.info("---- Desired/Status Stress Summary ----")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Operations: {total:,} (success={success:,}, failure={failure:,})")
        logger.info(f"Elapsed: {elapsed:.2f}s, Throughput: {tps:,.2f} ops/s")
        logger.info("---------------------------------------")

    def stop(self):
        self.running = False
        try:
            self._client.close()
        except Exception:
            pass


# ----------------------------- CLI -----------------------------


async def _amain(args):
    stresser = DesiredStatusStresser(
        etcd_host=args.host,
        etcd_port=args.port,
        mode=args.mode,
        fixed_node=args.fixed_node,
        fixed_depl=args.fixed_depl,
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
    p = argparse.ArgumentParser(description="Desired/Status store stress tester")
    p.add_argument("--host", type=str, default="localhost", help="etcd host")
    p.add_argument("--port", type=int, default=2379, help="etcd port")
    p.add_argument(
        "--mode",
        type=str,
        choices=[
            "desired_set",
            "desired_schedule",
            "status_claim",
            "status_set",
            "heartbeat",
            "end_to_end",
        ],
        default="end_to_end",
        help="stress mode",
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
    p.add_argument(
        "--fixed-node",
        type=str,
        default=None,
        help="use a fixed node instead of random",
    )
    p.add_argument(
        "--fixed-depl",
        type=str,
        default=None,
        help="use a fixed deployment instead of random",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    # Examples:
    # python -m tools.stress.state_store_stresser --mode end_to_end --run-time 30s --concurrency 32
    # python -m tools.stress.state_store_stresser --mode desired_set --num-requests 50000 --concurrency 64
    # python -m tools.stress.state_store_stresser --mode status_set --run-time 45s --fixed-node n1 --fixed-depl d1
    main()
