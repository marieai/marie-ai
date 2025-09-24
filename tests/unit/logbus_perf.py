"""
LogBus performance / correctness test.

What it measures
----------------
- Attempted records vs emitted vs dropped (from GLOBAL_LOG_BUS)
- End-to-end throughput (records/sec) under configurable load
- Tail-loss check after GLOBAL_LOG_BUS.stop() (should be zero: emitted == attempted - dropped)

Usage
-----
python logbus_perf.py \
  --threads 4 \
  --duration 5.0 \
  --rate-per-thread 20000 \
  --maxsize 50000 \
  --batch-size 256 \
  --flush-interval 0.02 \
  [--baseline-file baseline.log]  # optional direct FileHandler baseline

Tips
----
- For human terminal tests, keep sinks to a simple StreamHandler; RichHandler is pretty but slow.
- For max throughput tests, prefer a Null/Counting sink or a FileHandler on fast disk.
"""

import argparse
import logging
import threading
import time
from collections import deque
from typing import List, Optional

from marie.logging_core.log_bus import GLOBAL_LOG_BUS

# ---------- counting sink (thread-safe) ----------

class CountingHandler(logging.Handler):
    """A minimal sink that counts handle() calls and stores the last few messages (optional)."""

    def __init__(self, keep_last_n: int = 0):
        super().__init__()
        self._count = 0
        self._lock = threading.Lock()
        self._ring: Optional[deque] = deque(maxlen=keep_last_n) if keep_last_n > 0 else None

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._count += 1
            if self._ring is not None:
                try:
                    self._ring.append(record.getMessage())
                except Exception:
                    self._ring.append(str(record.msg))

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def last_messages(self) -> List[str]:
        with self._lock:
            return list(self._ring) if self._ring is not None else []


# ---------- workload ----------

def producer(logger: logging.Logger, stop_event: threading.Event, rate_per_sec: int, thread_id: int,
             attempts_counter: List[int]):
    """
    Fire-and-forget logging producer.
    Uses a simple pacing loop: try to keep ~rate_per_sec without busy-spinning too hard.
    """
    # time per log target
    target_dt = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
    next_t = time.monotonic()
    seq = 0
    while not stop_event.is_set():
        logger.info(f"perf_test tid={thread_id} seq={seq} sequence time : {time.monotonic()}")
        attempts_counter[0] += 1
        seq += 1

        if target_dt > 0:
            next_t += target_dt
            # short sleep to pace; don't over-sleep
            now = time.monotonic()
            delay = next_t - now
            if delay > 0:
                # sleep in small chunks to stay responsive
                if delay > 0.005:
                    time.sleep(0.002)
                else:
                    time.sleep(delay)
        else:
            # fully unpaced -> tiny yield
            time.sleep(0)


def run_bus_test(args) -> None:
    # 1) set bus tuning early (recreate singleton if your project allows; otherwise set at creation time)
    # For this test, we assume it's instantiated with desired defaults elsewhere.

    # 2) configure sinks
    sink = CountingHandler()
    sink.setLevel(logging.INFO)
    GLOBAL_LOG_BUS.set_sinks([sink])

    # 3) build a dedicated logger and attach to bus
    logger = logging.getLogger("perf.logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # avoid duplicate handlers if re-running
    for h in list(logger.handlers):
        logger.removeHandler(h)
    GLOBAL_LOG_BUS.attach_logger(logger)

    # 4) spin up producers
    stop_evt = threading.Event()
    attempts = [0]
    threads = []
    start = time.monotonic()
    for t_id in range(args.threads):
        th = threading.Thread(
            target=producer,
            args=(logger, stop_evt, args.rate_per_thread, t_id, attempts),
            name=f"prod-{t_id}",
            daemon=True,
        )
        th.start()
        threads.append(th)

    # 5) run for duration
    time.sleep(args.duration)
    stop_evt.set()
    for th in threads:
        th.join(timeout=2.0)

    # 6) flush + stop bus (final drain)
    t1 = time.monotonic()
    GLOBAL_LOG_BUS.stop(timeout=2.0)
    t2 = time.monotonic()

    # 7) results
    attempted = attempts[0]
    emitted = sink.count
    dropped = GLOBAL_LOG_BUS.dropped_count
    stop_ms = (t2 - t1) * 1000.0
    total_time = (t2 - start)

    print("\n=== LogBus PERF ===")
    print(f"threads           : {args.threads}")
    print(f"rate/thread       : {args.rate_per_thread}/s")
    print(f"duration          : {args.duration:.2f}s")
    print(f"queue maxsize     : {args.maxsize}  (see module config)")
    print(f"batch_size        : {args.batch_size}  (see module config)")
    print(f"flush_interval    : {args.flush_interval:.3f}s  (see module config)")
    print("---")
    print(f"attempted         : {attempted:,}")
    print(f"emitted (sinks)   : {emitted:,}")
    print(f"dropped (bus)     : {dropped:,}")
    print(f"stop() drain time : {stop_ms:.1f} ms")
    print(f"throughput (sink) : {emitted / total_time:,.0f} rec/s\n")

    # tail-loss check
    expected_emitted = attempted - dropped
    if emitted != expected_emitted:
        print(
            f"[WARN] tail-loss detected: emitted={emitted:,} expected={expected_emitted:,} (Δ={expected_emitted - emitted:,})")
    else:
        print("[OK] no tail-loss: emitted == attempted - dropped")


def run_baseline_file(args, filename: str) -> None:
    """
    Optional baseline without the bus: direct FileHandler.
    Useful to see formatter/disk cost independent from the bus.
    """
    logger = logging.getLogger("baseline.file")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # fresh handler
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(filename, mode="w", delay=False)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    stop_evt = threading.Event()
    attempts = [0]
    threads = []
    start = time.monotonic()
    for t_id in range(args.threads):
        th = threading.Thread(
            target=producer,
            args=(logger, stop_evt, args.rate_per_thread, t_id, attempts),
            name=f"base-prod-{t_id}",
            daemon=True,
        )
        th.start()
        threads.append(th)

    time.sleep(args.duration)
    stop_evt.set()
    for th in threads:
        th.join(timeout=2.0)

    t1 = time.monotonic()
    # ensure file is flushed
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    t2 = time.monotonic()

    attempted = attempts[0]
    stop_ms = (t2 - t1) * 1000.0
    total_time = (t2 - start)
    print("\n=== Baseline FileHandler ===")
    print(f"attempted         : {attempted:,}")
    print(f"stop() close time : {stop_ms:.1f} ms")
    print(f"throughput (approx): {attempted / total_time:,.0f} rec/s")
    print(f"output file       : {filename}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=4, help="producer threads")
    p.add_argument("--duration", type=float, default=5.0, help="test duration seconds")
    p.add_argument("--rate-per-thread", type=int, default=20000, help="target logs/sec per thread")
    # the next three are informational here; set them where GLOBAL_LOG_BUS is constructed
    p.add_argument("--maxsize", type=int, default=10000, help="queue maxsize (informational)")
    p.add_argument("--batch-size", type=int, default=512, help="batch size (informational)")
    p.add_argument("--flush-interval", type=float, default=0.03, help="flush interval seconds (informational)")
    p.add_argument("--baseline-file", type=str, default="", help="optional: run baseline with direct FileHandler")
    args = p.parse_args()

    run_bus_test(args)
    if args.baseline_file:
        run_baseline_file(args, args.baseline_file)


if __name__ == "__main__":
    main()
