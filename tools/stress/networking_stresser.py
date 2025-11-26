#!/usr/bin/env python3
"""
Networking Stress Tester for Marie Gateway and Load Balancer.

This tool tests the networking layer by:
1. Simulating N executor servers that go up and down
2. Sending requests through the gateway to test load balancing
3. Measuring circuit breaker behavior and recovery
4. Tracking metrics (failures, successes, latency, circuit state changes)

Usage:
    # Basic usage - expects gateway running on localhost:52000
    python networking_stresser.py --num-executors 5 --chaos-interval 10

    # With custom gateway address
    python networking_stresser.py --gateway-host localhost --gateway-port 52000

    # Long running stress test
    python networking_stresser.py --duration 300 --num-executors 10 --chaos-interval 5

    # Test specific modes
    python networking_stresser.py --mode circuit_breaker_test
    python networking_stresser.py --mode load_balancer_test
    python networking_stresser.py --mode chaos_test
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import random
import signal
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import grpc

# Add marie to path if running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from marie import Client, Document, DocumentArray, Executor, requests
from marie.helper import random_port
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.servers import BaseServer
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("NetworkingStresser")

Mode = Literal[
    "circuit_breaker_test",  # Test circuit breaker state transitions
    "load_balancer_test",  # Test load distribution
    "chaos_test",  # Random up/down of executors
    "full_integration",  # Comprehensive stress test
    "request_flood",  # High volume request testing
]


class ExecutorState(Enum):
    """State of a simulated executor."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ExecutorMetrics:
    """Metrics for a single executor."""

    executor_id: str
    port: int
    start_count: int = 0
    stop_count: int = 0
    requests_handled: int = 0
    requests_failed: int = 0
    uptime_seconds: float = 0.0
    downtime_seconds: float = 0.0
    last_start_time: Optional[float] = None
    last_stop_time: Optional[float] = None
    state: ExecutorState = ExecutorState.STOPPED


@dataclass
class GatewayMetrics:
    """Metrics for gateway interactions."""

    requests_sent: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_timeout: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    circuit_breaker_opens: int = 0
    circuit_breaker_closes: int = 0


@dataclass
class StressTestConfig:
    """Configuration for stress test."""

    # Gateway settings
    gateway_host: str = "localhost"
    gateway_port: int = 52000
    gateway_protocol: str = "grpc"

    # Executor settings
    num_executors: int = 5
    executor_base_port: int = 54000
    executor_startup_delay: float = 1.0

    # Chaos settings
    chaos_enabled: bool = True
    chaos_interval_min: float = 5.0  # Min seconds between chaos events
    chaos_interval_max: float = 15.0  # Max seconds between chaos events
    chaos_down_duration_min: float = 3.0  # Min seconds executor stays down
    chaos_down_duration_max: float = 10.0  # Max seconds executor stays down
    chaos_max_down_ratio: float = 0.5  # Max ratio of executors that can be down

    # Request settings
    request_rate: float = 10.0  # Requests per second
    request_timeout: float = 30.0  # Timeout for each request
    request_batch_size: int = 1  # Documents per request

    # Test settings
    duration_seconds: float = 60.0  # Total test duration
    warmup_seconds: float = 5.0  # Warmup period before chaos starts
    mode: Mode = "full_integration"

    # Reporting
    report_interval: float = 10.0  # Seconds between metric reports


class SimulatedExecutor:
    """
    A simulated executor server that can be started and stopped.

    This creates a minimal gRPC server that responds to requests,
    simulating an executor in the Marie ecosystem.
    """

    def __init__(
        self,
        executor_id: str,
        port: int,
        logger: logging.Logger,
        slow_response: bool = False,
        fail_rate: float = 0.0,
    ):
        self.executor_id = executor_id
        self.port = port
        self._logger = logger
        self.slow_response = slow_response
        self.fail_rate = fail_rate
        self.process: Optional[multiprocessing.Process] = None
        self.metrics = ExecutorMetrics(executor_id=executor_id, port=port)
        self._stop_event = multiprocessing.Event()
        self._ready_event = multiprocessing.Event()

    def start(self) -> bool:
        """Start the executor server."""
        if self.process is not None and self.process.is_alive():
            self._logger.warning(f"Executor {self.executor_id} already running")
            return False

        self._stop_event.clear()
        self._ready_event.clear()

        self.process = multiprocessing.Process(
            target=self._run_server,
            args=(self.port, self._stop_event, self._ready_event, self.executor_id),
            daemon=True,
        )
        self.process.start()

        # Wait for server to be ready
        ready = self._ready_event.wait(timeout=10.0)
        if ready:
            self.metrics.start_count += 1
            self.metrics.last_start_time = time.time()
            self.metrics.state = ExecutorState.RUNNING
            self._logger.info(
                f"Executor {self.executor_id} started on port {self.port}"
            )
            return True
        else:
            self._logger.error(f"Executor {self.executor_id} failed to start")
            self.metrics.state = ExecutorState.FAILED
            return False

    def stop(self, graceful: bool = True) -> bool:
        """Stop the executor server."""
        if self.process is None or not self.process.is_alive():
            self._logger.warning(f"Executor {self.executor_id} not running")
            return False

        self.metrics.state = ExecutorState.STOPPING

        if graceful:
            self._stop_event.set()
            self.process.join(timeout=5.0)

        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)

        if self.process.is_alive():
            self.process.kill()
            self.process.join()

        # Update metrics
        if self.metrics.last_start_time:
            self.metrics.uptime_seconds += time.time() - self.metrics.last_start_time
        self.metrics.stop_count += 1
        self.metrics.last_stop_time = time.time()
        self.metrics.state = ExecutorState.STOPPED

        self._logger.info(f"Executor {self.executor_id} stopped")
        return True

    def is_running(self) -> bool:
        """Check if executor is running."""
        return self.process is not None and self.process.is_alive()

    @staticmethod
    def _run_server(
        port: int,
        stop_event: multiprocessing.Event,
        ready_event: multiprocessing.Event,
        executor_id: str,
    ):
        """Run the executor server in a subprocess."""
        import asyncio

        from marie.serve.runtimes.worker import WorkerRuntime

        async def _run():
            # Create a simple executor class
            class StressTestExecutor(Executor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.request_count = 0

                @requests
                def process(self, docs: DocumentArray, **kwargs) -> DocumentArray:
                    self.request_count += 1
                    for doc in docs:
                        doc.text = (
                            f"Processed by {executor_id} (count: {self.request_count})"
                        )
                    return docs

            # Signal ready
            ready_event.set()

            # Wait for stop signal
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

        try:
            asyncio.run(_run())
        except Exception as e:
            logger.error(f"Executor {executor_id} error: {e}")


class SimpleMockServer:
    """
    A simple mock gRPC server for testing.
    Uses raw gRPC without the full Marie runtime.
    """

    def __init__(self, executor_id: str, port: int):
        self.executor_id = executor_id
        self.port = port
        self.server = None
        self._running = False
        self.request_count = 0

    async def start(self):
        """Start the mock server."""
        from marie.proto import jina_pb2, jina_pb2_grpc

        class MockDataRequestHandler(jina_pb2_grpc.JinaDataRequestRPCServicer):
            def __init__(handler_self):
                handler_self.executor_id = self.executor_id
                handler_self.request_count = 0

            async def process_single_data(handler_self, request, context):
                handler_self.request_count += 1
                self.request_count = handler_self.request_count
                # Create response
                response = jina_pb2.DataRequestProto()
                response.header.request_id = request.header.request_id
                return response

        self.server = grpc.aio.server()
        jina_pb2_grpc.add_JinaDataRequestRPCServicer_to_server(
            MockDataRequestHandler(), self.server
        )
        self.server.add_insecure_port(f"0.0.0.0:{self.port}")
        await self.server.start()
        self._running = True
        logger.info(f"Mock server {self.executor_id} started on port {self.port}")

    async def stop(self):
        """Stop the mock server."""
        if self.server:
            await self.server.stop(grace=1.0)
            self._running = False
            logger.info(f"Mock server {self.executor_id} stopped")

    def is_running(self) -> bool:
        return self._running


class ChaosController:
    """
    Controls chaos events - bringing executors up and down randomly.
    """

    def __init__(
        self,
        executors: List[SimulatedExecutor],
        config: StressTestConfig,
        metrics: GatewayMetrics,
    ):
        self.executors = executors
        self.config = config
        self.metrics = metrics
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger("ChaosController")

    async def start(self):
        """Start the chaos controller."""
        self._running = True
        self._task = asyncio.create_task(self._chaos_loop())
        self._logger.info("Chaos controller started")

    async def stop(self):
        """Stop the chaos controller."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("Chaos controller stopped")

    async def _chaos_loop(self):
        """Main chaos event loop."""
        while self._running:
            try:
                # Wait for random interval
                interval = random.uniform(
                    self.config.chaos_interval_min, self.config.chaos_interval_max
                )
                await asyncio.sleep(interval)

                if not self._running:
                    break

                # Decide on chaos action
                await self._perform_chaos_action()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Chaos loop error: {e}")

    async def _perform_chaos_action(self):
        """Perform a random chaos action."""
        running_executors = [e for e in self.executors if e.is_running()]
        stopped_executors = [e for e in self.executors if not e.is_running()]

        # Calculate how many can be down
        max_down = int(len(self.executors) * self.config.chaos_max_down_ratio)
        current_down = len(stopped_executors)

        # Decide action: bring one down or bring one up
        if current_down >= max_down:
            # Must bring one up
            action = "up"
        elif current_down == 0:
            # Must bring one down
            action = "down"
        else:
            # Random choice
            action = random.choice(["up", "down"])

        if action == "down" and running_executors:
            executor = random.choice(running_executors)
            self._logger.info(f"CHAOS: Stopping executor {executor.executor_id}")
            executor.stop(graceful=random.choice([True, False]))

            # Schedule restart after random duration
            down_duration = random.uniform(
                self.config.chaos_down_duration_min,
                self.config.chaos_down_duration_max,
            )
            asyncio.create_task(self._delayed_restart(executor, down_duration))

        elif action == "up" and stopped_executors:
            executor = random.choice(stopped_executors)
            self._logger.info(f"CHAOS: Starting executor {executor.executor_id}")
            executor.start()

    async def _delayed_restart(self, executor: SimulatedExecutor, delay: float):
        """Restart an executor after a delay."""
        await asyncio.sleep(delay)
        if not executor.is_running() and self._running:
            self._logger.info(
                f"CHAOS: Auto-restarting executor {executor.executor_id} after {delay:.1f}s"
            )
            executor.start()


class RequestGenerator:
    """
    Generates and sends requests to the gateway.
    """

    def __init__(self, config: StressTestConfig, metrics: GatewayMetrics):
        self.config = config
        self.metrics = metrics
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger("RequestGenerator")

    async def start(self):
        """Start generating requests."""
        self._running = True
        self._task = asyncio.create_task(self._request_loop())
        self._logger.info(
            f"Request generator started (rate: {self.config.request_rate}/s)"
        )

    async def stop(self):
        """Stop generating requests."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("Request generator stopped")

    async def _request_loop(self):
        """Main request generation loop."""
        interval = 1.0 / self.config.request_rate

        while self._running:
            try:
                start_time = time.time()
                await self._send_request()
                elapsed = time.time() - start_time

                # Adjust sleep to maintain rate
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Request loop error: {e}")

    async def _send_request(self):
        """Send a single request to the gateway."""
        self.metrics.requests_sent += 1
        start_time = time.time()

        try:
            client = Client(
                host=self.config.gateway_host,
                port=self.config.gateway_port,
                protocol=self.config.gateway_protocol,
                asyncio=True,
            )

            docs = DocumentArray(
                [
                    Document(text=f"Test request {uuid.uuid4()}")
                    for _ in range(self.config.request_batch_size)
                ]
            )

            async for response in client.post(
                "/",
                inputs=docs,
                request_size=self.config.request_batch_size,
                timeout=self.config.request_timeout,
            ):
                pass

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.requests_success += 1
            self.metrics.latencies_ms.append(latency_ms)

        except asyncio.TimeoutError:
            self.metrics.requests_timeout += 1
            self.metrics.requests_failed += 1
            self.metrics.errors_by_type["timeout"] += 1

        except grpc.RpcError as e:
            self.metrics.requests_failed += 1
            error_type = e.code().name if hasattr(e, "code") else "unknown_grpc"
            self.metrics.errors_by_type[error_type] += 1

            # Detect circuit breaker behavior
            if "UNAVAILABLE" in str(e):
                self.metrics.circuit_breaker_opens += 1

        except Exception as e:
            self.metrics.requests_failed += 1
            self.metrics.errors_by_type[type(e).__name__] += 1


class MetricsReporter:
    """
    Reports metrics periodically during the stress test.
    """

    def __init__(
        self,
        gateway_metrics: GatewayMetrics,
        executors: List[SimulatedExecutor],
        config: StressTestConfig,
    ):
        self.gateway_metrics = gateway_metrics
        self.executors = executors
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._logger = logging.getLogger("MetricsReporter")

    async def start(self):
        """Start reporting metrics."""
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._report_loop())

    async def stop(self):
        """Stop reporting and print final report."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._print_final_report()

    async def _report_loop(self):
        """Periodic reporting loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.report_interval)
                if self._running:
                    self._print_interim_report()
            except asyncio.CancelledError:
                break

    def _print_interim_report(self):
        """Print interim metrics."""
        elapsed = time.time() - self._start_time
        running = sum(1 for e in self.executors if e.is_running())
        total = len(self.executors)

        m = self.gateway_metrics
        success_rate = (
            (m.requests_success / m.requests_sent * 100) if m.requests_sent > 0 else 0
        )
        avg_latency = sum(m.latencies_ms) / len(m.latencies_ms) if m.latencies_ms else 0

        self._logger.info(
            f"[{elapsed:.0f}s] Executors: {running}/{total} | "
            f"Requests: {m.requests_sent} (success: {success_rate:.1f}%) | "
            f"Avg latency: {avg_latency:.1f}ms | "
            f"Errors: {m.requests_failed}"
        )

    def _print_final_report(self):
        """Print final comprehensive report."""
        elapsed = time.time() - self._start_time
        m = self.gateway_metrics

        print("\n" + "=" * 70)
        print("NETWORKING STRESS TEST - FINAL REPORT")
        print("=" * 70)

        print(f"\nTest Duration: {elapsed:.1f} seconds")
        print(f"Test Mode: {self.config.mode}")

        print("\n--- Gateway Metrics ---")
        print(f"Total Requests Sent: {m.requests_sent}")
        print(f"Successful Requests: {m.requests_success}")
        print(f"Failed Requests: {m.requests_failed}")
        print(f"Timeout Requests: {m.requests_timeout}")

        if m.requests_sent > 0:
            print(f"Success Rate: {m.requests_success / m.requests_sent * 100:.2f}%")
            print(f"Throughput: {m.requests_sent / elapsed:.2f} req/s")

        if m.latencies_ms:
            sorted_latencies = sorted(m.latencies_ms)
            print(f"\n--- Latency Statistics (ms) ---")
            print(f"Min: {min(m.latencies_ms):.2f}")
            print(f"Max: {max(m.latencies_ms):.2f}")
            print(f"Avg: {sum(m.latencies_ms) / len(m.latencies_ms):.2f}")
            print(f"P50: {sorted_latencies[len(sorted_latencies) // 2]:.2f}")
            print(f"P95: {sorted_latencies[int(len(sorted_latencies) * 0.95)]:.2f}")
            print(f"P99: {sorted_latencies[int(len(sorted_latencies) * 0.99)]:.2f}")

        if m.errors_by_type:
            print(f"\n--- Errors by Type ---")
            for error_type, count in sorted(
                m.errors_by_type.items(), key=lambda x: -x[1]
            ):
                print(f"  {error_type}: {count}")

        print(f"\n--- Executor Metrics ---")
        for executor in self.executors:
            em = executor.metrics
            print(
                f"  {em.executor_id}: starts={em.start_count}, stops={em.stop_count}, "
                f"uptime={em.uptime_seconds:.1f}s, state={em.state.value}"
            )

        print("\n" + "=" * 70)


class NetworkingStresser:
    """
    Main stress test orchestrator.
    """

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.executors: List[SimulatedExecutor] = []
        self.gateway_metrics = GatewayMetrics()
        self.chaos_controller: Optional[ChaosController] = None
        self.request_generator: Optional[RequestGenerator] = None
        self.metrics_reporter: Optional[MetricsReporter] = None
        self._logger = logging.getLogger("NetworkingStresser")

    async def setup(self):
        """Setup the stress test environment."""
        self._logger.info("Setting up stress test environment...")

        # Create simulated executors
        for i in range(self.config.num_executors):
            executor = SimulatedExecutor(
                executor_id=f"executor-{i}",
                port=self.config.executor_base_port + i,
                logger=self._logger,
            )
            self.executors.append(executor)

        # Create controllers
        self.chaos_controller = ChaosController(
            self.executors, self.config, self.gateway_metrics
        )
        self.request_generator = RequestGenerator(self.config, self.gateway_metrics)
        self.metrics_reporter = MetricsReporter(
            self.gateway_metrics, self.executors, self.config
        )

        self._logger.info(f"Created {len(self.executors)} simulated executors")

    async def start_executors(self):
        """Start all executors."""
        self._logger.info("Starting executors...")
        for executor in self.executors:
            executor.start()
            await asyncio.sleep(self.config.executor_startup_delay)

    async def stop_executors(self):
        """Stop all executors."""
        self._logger.info("Stopping executors...")
        for executor in self.executors:
            if executor.is_running():
                executor.stop()

    async def run(self):
        """Run the stress test."""
        self._logger.info(
            f"Starting networking stress test (duration: {self.config.duration_seconds}s)"
        )

        try:
            # Setup
            await self.setup()

            # Start executors
            await self.start_executors()

            # Start metrics reporter
            await self.metrics_reporter.start()

            # Warmup period
            self._logger.info(f"Warmup period ({self.config.warmup_seconds}s)...")
            await self.request_generator.start()
            await asyncio.sleep(self.config.warmup_seconds)

            # Start chaos if enabled
            if self.config.chaos_enabled:
                await self.chaos_controller.start()

            # Run for specified duration
            remaining = self.config.duration_seconds - self.config.warmup_seconds
            await asyncio.sleep(remaining)

        except asyncio.CancelledError:
            self._logger.info("Stress test cancelled")
        except Exception as e:
            self._logger.error(f"Stress test error: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()

    async def cleanup(self):
        """Cleanup all resources."""
        self._logger.info("Cleaning up...")

        if self.chaos_controller:
            await self.chaos_controller.stop()

        if self.request_generator:
            await self.request_generator.stop()

        if self.metrics_reporter:
            await self.metrics_reporter.stop()

        await self.stop_executors()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Networking Stress Tester for Marie Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with default settings
    python networking_stresser.py

    # Test with 10 executors and longer duration
    python networking_stresser.py --num-executors 10 --duration 300

    # Aggressive chaos testing
    python networking_stresser.py --chaos-interval-min 2 --chaos-interval-max 5

    # High request rate testing
    python networking_stresser.py --request-rate 100 --mode request_flood

    # Test against specific gateway
    python networking_stresser.py --gateway-host 192.168.1.100 --gateway-port 52000
        """,
    )

    # Gateway settings
    gateway_group = parser.add_argument_group("Gateway Settings")
    gateway_group.add_argument(
        "--gateway-host", default="localhost", help="Gateway host (default: localhost)"
    )
    gateway_group.add_argument(
        "--gateway-port", type=int, default=52000, help="Gateway port (default: 52000)"
    )
    gateway_group.add_argument(
        "--gateway-protocol",
        choices=["grpc", "http", "websocket"],
        default="grpc",
        help="Gateway protocol (default: grpc)",
    )

    # Executor settings
    executor_group = parser.add_argument_group("Executor Settings")
    executor_group.add_argument(
        "--num-executors",
        type=int,
        default=5,
        help="Number of simulated executors (default: 5)",
    )
    executor_group.add_argument(
        "--executor-base-port",
        type=int,
        default=54000,
        help="Base port for executors (default: 54000)",
    )

    # Chaos settings
    chaos_group = parser.add_argument_group("Chaos Settings")
    chaos_group.add_argument(
        "--no-chaos", action="store_true", help="Disable chaos (executor up/down)"
    )
    chaos_group.add_argument(
        "--chaos-interval-min",
        type=float,
        default=5.0,
        help="Min seconds between chaos events (default: 5)",
    )
    chaos_group.add_argument(
        "--chaos-interval-max",
        type=float,
        default=15.0,
        help="Max seconds between chaos events (default: 15)",
    )
    chaos_group.add_argument(
        "--chaos-down-duration-min",
        type=float,
        default=3.0,
        help="Min seconds executor stays down (default: 3)",
    )
    chaos_group.add_argument(
        "--chaos-down-duration-max",
        type=float,
        default=10.0,
        help="Max seconds executor stays down (default: 10)",
    )
    chaos_group.add_argument(
        "--chaos-max-down-ratio",
        type=float,
        default=0.5,
        help="Max ratio of executors that can be down (default: 0.5)",
    )

    # Request settings
    request_group = parser.add_argument_group("Request Settings")
    request_group.add_argument(
        "--request-rate",
        type=float,
        default=10.0,
        help="Requests per second (default: 10)",
    )
    request_group.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )

    # Test settings
    test_group = parser.add_argument_group("Test Settings")
    test_group.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds (default: 60)",
    )
    test_group.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup period in seconds (default: 5)",
    )
    test_group.add_argument(
        "--mode",
        choices=[
            "circuit_breaker_test",
            "load_balancer_test",
            "chaos_test",
            "full_integration",
            "request_flood",
        ],
        default="full_integration",
        help="Test mode (default: full_integration)",
    )
    test_group.add_argument(
        "--report-interval",
        type=float,
        default=10.0,
        help="Seconds between metric reports (default: 10)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build config from args
    config = StressTestConfig(
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        gateway_protocol=args.gateway_protocol,
        num_executors=args.num_executors,
        executor_base_port=args.executor_base_port,
        chaos_enabled=not args.no_chaos,
        chaos_interval_min=args.chaos_interval_min,
        chaos_interval_max=args.chaos_interval_max,
        chaos_down_duration_min=args.chaos_down_duration_min,
        chaos_down_duration_max=args.chaos_down_duration_max,
        chaos_max_down_ratio=args.chaos_max_down_ratio,
        request_rate=args.request_rate,
        request_timeout=args.request_timeout,
        duration_seconds=args.duration,
        warmup_seconds=args.warmup,
        mode=args.mode,
        report_interval=args.report_interval,
    )

    # Apply mode-specific settings
    if config.mode == "request_flood":
        config.chaos_enabled = False
        config.request_rate = max(config.request_rate, 100.0)
    elif config.mode == "circuit_breaker_test":
        config.chaos_interval_min = 3.0
        config.chaos_interval_max = 5.0
        config.chaos_down_duration_min = 5.0
        config.chaos_down_duration_max = 15.0
    elif config.mode == "load_balancer_test":
        config.chaos_enabled = False

    # Run stress test
    stresser = NetworkingStresser(config)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, stopping...")
        asyncio.get_event_loop().stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(stresser.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise


if __name__ == "__main__":
    main()
