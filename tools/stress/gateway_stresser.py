#!/usr/bin/env python3
"""
Gateway Stress Tester for Marie-AI.

This is a simpler stress tester that:
1. Connects to an already running Marie gateway
2. Sends requests at configurable rates and concurrency
3. Measures latency, success rates, throughput
4. Tests circuit breaker and load balancer behavior

Prerequisites:
    - Marie gateway must be running (started via `marie server --start ...`)
    - Executors should be running (can be started/stopped externally to test chaos)

Usage:
    # Basic stress test against running gateway
    python gateway_stresser.py --gateway-port 52000 --duration 60

    # High concurrency test
    python gateway_stresser.py --concurrency 50 --request-rate 100

    # Test specific endpoint
    python gateway_stresser.py --endpoint /extract --duration 120

    # HTTP protocol test
    python gateway_stresser.py --protocol http --gateway-port 52001

Examples:
    # Start gateway in another terminal:
    # marie server --start --uses config/service/marie.yml

    # Then run this stress tester:
    python gateway_stresser.py --duration 60 --request-rate 20
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add marie to path if running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import aiohttp
    import grpc

    from marie import Client, Document, DocumentArray
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Make sure marie and aiohttp are installed:")
    print("  pip install aiohttp")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("GatewayStresser")


@dataclass
class RequestResult:
    """Result of a single request."""

    request_id: str
    success: bool
    latency_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class StressMetrics:
    """Aggregated stress test metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0

    latencies_ms: List[float] = field(default_factory=list)
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_messages: List[str] = field(default_factory=list)

    # Time series data for charts
    requests_per_second: List[Tuple[float, int]] = field(default_factory=list)
    latencies_over_time: List[Tuple[float, float]] = field(default_factory=list)

    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.latencies_ms.append(latency_ms)

    def record_failure(self, error_type: str, error_message: str = ""):
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.errors_by_type[error_type] += 1
        if error_message and len(self.errors_messages) < 100:  # Limit stored messages
            self.errors_messages.append(f"{error_type}: {error_message[:200]}")

    def record_timeout(self):
        """Record a timeout."""
        self.total_requests += 1
        self.timeout_requests += 1
        self.failed_requests += 1
        self.errors_by_type["TIMEOUT"] += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        """Calculate P50 latency."""
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        """Calculate P95 latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        """Calculate P99 latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def throughput(self) -> float:
        """Calculate throughput (requests per second)."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.total_requests / duration


class GatewayStresser:
    """
    Stress tester for Marie gateway.

    Sends concurrent requests to the gateway and collects metrics.
    Supports both gRPC and HTTP protocols.
    """

    def __init__(
        self,
        gateway_host: str = "localhost",
        gateway_port: int = 52000,
        http_port: Optional[int] = None,
        protocol: str = "grpc",
        endpoint: str = "/api/v1/invoke",
        concurrency: int = 10,
        request_rate: float = 10.0,
        timeout: float = 30.0,
        duration: float = 60.0,
        warmup: float = 5.0,
        document_text: str = "Test document for stress testing",
        batch_size: int = 1,
        http_method: str = "POST",
        use_direct_http: bool = True,
        api_key: Optional[str] = None,
        request_parameters: Optional[Dict[str, Any]] = None,
        target_executor: Optional[str] = None,
    ):
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.http_port = http_port or gateway_port
        self.protocol = protocol
        self.endpoint = endpoint
        self.concurrency = concurrency
        self.request_rate = request_rate
        self.timeout = timeout
        self.duration = duration
        self.warmup = warmup
        self.document_text = document_text
        self.batch_size = batch_size
        self.http_method = http_method.upper()
        self.use_direct_http = use_direct_http  # Use aiohttp directly for HTTP
        self.api_key = api_key
        self.request_parameters = request_parameters or {}
        self.target_executor = target_executor

        self.metrics = StressMetrics()
        self._running = False
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    async def _send_request(self, request_id: str) -> RequestResult:
        """Send a single request to the gateway using the appropriate protocol."""
        if self.protocol == "http" and self.use_direct_http:
            return await self._send_http_request(request_id)
        else:
            return await self._send_grpc_request(request_id)

    async def _send_http_request(self, request_id: str) -> RequestResult:
        """Send a single HTTP request to the gateway using aiohttp."""
        start_time = time.time()

        try:
            # Build the URL
            base_url = f"http://{self.gateway_host}:{self.http_port}"
            endpoint = (
                self.endpoint if self.endpoint.startswith("/") else f"/{self.endpoint}"
            )
            url = f"{base_url}{endpoint}"

            # Build the request payload (Marie gateway format)
            # Matches the format expected by http_fastapi_app_docarrayv2.py
            docs_data = [
                {
                    "id": f"{request_id}-{i}",
                    "text": f"{self.document_text} - {request_id} - {i}",
                }
                for i in range(self.batch_size)
            ]

            # Marie gateway expects: data, parameters, header (optional)
            payload = {
                "data": docs_data,
                "parameters": self.request_parameters or {},
                "header": {
                    "requestId": request_id,
                    "targetExecutor": self.target_executor or "",
                },
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Add authorization header if API key provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with self._http_session.request(
                self.http_method,
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                response_text = await response.text()
                latency_ms = (time.time() - start_time) * 1000

                if response.status >= 200 and response.status < 300:
                    return RequestResult(
                        request_id=request_id,
                        success=True,
                        latency_ms=latency_ms,
                    )
                else:
                    return RequestResult(
                        request_id=request_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type=f"HTTP_{response.status}",
                        error_message=response_text[:200],
                    )

        except asyncio.TimeoutError:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type="TIMEOUT",
                error_message="HTTP request timed out",
            )

        except aiohttp.ClientConnectorError as e:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type="CONNECTION_ERROR",
                error_message=str(e)[:200],
            )

        except aiohttp.ClientError as e:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type=f"HTTP_{type(e).__name__}",
                error_message=str(e)[:200],
            )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
            )

    async def _send_grpc_request(self, request_id: str) -> RequestResult:
        """Send a single gRPC request to the gateway using Marie Client."""
        start_time = time.time()

        try:
            client = Client(
                host=self.gateway_host,
                port=self.gateway_port,
                protocol=self.protocol,
                asyncio=True,
            )

            docs = DocumentArray(
                [
                    Document(text=f"{self.document_text} - {request_id} - {i}")
                    for i in range(self.batch_size)
                ]
            )

            # Build request kwargs (headers for auth)
            request_kwargs = {}
            if self.api_key:
                request_kwargs["headers"] = [
                    ("Authorization", f"Bearer {self.api_key}")
                ]

            response_docs = None
            async for response in client.post(
                self.endpoint,
                inputs=docs,
                parameters=self.request_parameters,
                request_size=self.batch_size,
                timeout=self.timeout,
                **request_kwargs,
            ):
                response_docs = response

            latency_ms = (time.time() - start_time) * 1000

            return RequestResult(
                request_id=request_id,
                success=True,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type="TIMEOUT",
                error_message="Request timed out",
            )

        except grpc.RpcError as e:
            error_code = e.code().name if hasattr(e, "code") else "UNKNOWN"
            error_details = e.details() if hasattr(e, "details") else str(e)
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type=f"GRPC_{error_code}",
                error_message=error_details[:200],
            )

        except ConnectionRefusedError:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type="CONNECTION_REFUSED",
                error_message="Could not connect to gateway",
            )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
            )

    async def _test_connectivity(self) -> bool:
        """Test gateway connectivity using health endpoint."""
        if self.protocol == "http" and self.use_direct_http:
            return await self._test_http_connectivity()
        else:
            # For gRPC, try a simple request
            test_result = await self._send_grpc_request("connectivity-test")
            if not test_result.success:
                self._logger.error(
                    f"Cannot connect to gateway: {test_result.error_type} - {test_result.error_message}"
                )
                return False
            return True

    async def _test_http_connectivity(self) -> bool:
        """Test HTTP gateway connectivity using /status or /dry_run endpoint."""
        base_url = f"http://{self.gateway_host}:{self.http_port}"

        # Try multiple health endpoints in order
        health_endpoints = ["/status", "/dry_run", "/"]

        for endpoint in health_endpoints:
            url = f"{base_url}{endpoint}"
            try:
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with self._http_session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        self._logger.info(f"Health check passed on {endpoint}")
                        return True
                    elif response.status == 404:
                        # Endpoint not found, try next
                        continue
                    else:
                        # Other status, might still be reachable
                        self._logger.debug(
                            f"Health endpoint {endpoint} returned {response.status}"
                        )
                        continue
            except aiohttp.ClientConnectorError as e:
                self._logger.error(f"Cannot connect to gateway: {e}")
                return False
            except Exception as e:
                self._logger.debug(f"Health check on {endpoint} failed: {e}")
                continue

        # If no health endpoint worked, try a simple connection test
        self._logger.warning(
            "No standard health endpoint available, testing connection with target endpoint..."
        )

        # Try the actual endpoint with a test request
        test_result = await self._send_http_request("connectivity-test")
        if test_result.success:
            return True

        # Check if it's a real connection error vs endpoint error
        if test_result.error_type in ["CONNECTION_ERROR", "TIMEOUT"]:
            self._logger.error(
                f"Cannot connect to gateway: {test_result.error_type} - {test_result.error_message}"
            )
            return False

        # 4xx/5xx errors mean gateway is reachable but endpoint might be wrong
        if test_result.error_type.startswith(
            "HTTP_4"
        ) or test_result.error_type.startswith("HTTP_5"):
            self._logger.warning(
                f"Gateway reachable but endpoint returned error: {test_result.error_type}"
            )
            self._logger.warning(
                f"Proceeding with stress test - errors may indicate endpoint configuration issues"
            )
            return True

        self._logger.error(
            f"Cannot connect to gateway: {test_result.error_type} - {test_result.error_message}"
        )
        return False

    async def _request_worker(self, worker_id: int):
        """Worker coroutine that sends requests at the specified rate."""
        interval = (
            self.concurrency / self.request_rate
        )  # Time between requests per worker
        request_count = 0

        while self._running:
            try:
                async with self._semaphore:
                    request_id = f"w{worker_id}-r{request_count}"
                    result = await self._send_request(request_id)

                    # Record metrics
                    if result.success:
                        self.metrics.record_success(result.latency_ms)
                    elif result.error_type == "TIMEOUT":
                        self.metrics.record_timeout()
                    else:
                        self.metrics.record_failure(
                            result.error_type, result.error_message or ""
                        )

                    request_count += 1

                # Rate limiting
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

    async def _progress_reporter(self, report_interval: float = 5.0):
        """Report progress periodically."""
        last_total = 0

        while self._running:
            await asyncio.sleep(report_interval)

            if not self._running:
                break

            elapsed = time.time() - self.metrics.start_time
            current_total = self.metrics.total_requests
            requests_in_interval = current_total - last_total
            rate = requests_in_interval / report_interval

            self._logger.info(
                f"[{elapsed:.0f}s] Requests: {current_total} | "
                f"Success: {self.metrics.success_rate:.1f}% | "
                f"Rate: {rate:.1f}/s | "
                f"Avg latency: {self.metrics.avg_latency_ms:.1f}ms | "
                f"Failed: {self.metrics.failed_requests}"
            )

            last_total = current_total

    async def run(self) -> StressMetrics:
        """Run the stress test."""
        # Display port info based on protocol
        if self.protocol == "http":
            port_info = f"HTTP port: {self.http_port}"
        else:
            port_info = f"gRPC port: {self.gateway_port}"

        self._logger.info(
            f"Starting gateway stress test:\n"
            f"  Gateway: {self.gateway_host} ({self.protocol})\n"
            f"  {port_info}\n"
            f"  Endpoint: {self.endpoint}\n"
            f"  Concurrency: {self.concurrency}\n"
            f"  Target rate: {self.request_rate} req/s\n"
            f"  Duration: {self.duration}s (warmup: {self.warmup}s)\n"
            f"  Timeout: {self.timeout}s"
        )

        # Create HTTP session if needed
        if self.protocol == "http" and self.use_direct_http:
            connector = aiohttp.TCPConnector(
                limit=self.concurrency * 2,
                limit_per_host=self.concurrency * 2,
                enable_cleanup_closed=True,
            )
            self._http_session = aiohttp.ClientSession(connector=connector)
            self._logger.info("HTTP session created with aiohttp")

        try:
            # Test connectivity first using health endpoint
            self._logger.info("Testing gateway connectivity...")
            connectivity_ok = await self._test_connectivity()
            if not connectivity_ok:
                if self.protocol == "http":
                    self._logger.error(
                        "Make sure the gateway is running and accessible at "
                        f"http://{self.gateway_host}:{self.http_port}"
                    )
                else:
                    self._logger.error(
                        "Make sure the gateway is running and accessible at "
                        f"{self.gateway_host}:{self.gateway_port}"
                    )
                return self.metrics

            self._logger.info("Gateway connectivity OK")

            # Initialize
            self._running = True
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self.metrics.start_time = time.time()

            # Start workers
            workers = [
                asyncio.create_task(self._request_worker(i))
                for i in range(self.concurrency)
            ]

            # Start progress reporter
            reporter = asyncio.create_task(self._progress_reporter())

            # Warmup
            if self.warmup > 0:
                self._logger.info(f"Warmup period ({self.warmup}s)...")
                await asyncio.sleep(self.warmup)
                # Reset metrics after warmup
                self.metrics = StressMetrics()
                self.metrics.start_time = time.time()
                self._logger.info("Warmup complete, starting measurement...")

            # Run for duration
            try:
                await asyncio.sleep(self.duration - self.warmup)
            except asyncio.CancelledError:
                self._logger.info("Test cancelled")

            # Stop
            self._running = False
            self.metrics.end_time = time.time()

            # Cancel workers
            for worker in workers:
                worker.cancel()
            reporter.cancel()

            await asyncio.gather(*workers, reporter, return_exceptions=True)

        finally:
            # Close HTTP session
            if self._http_session is not None:
                await self._http_session.close()
                self._http_session = None

        return self.metrics

    def print_report(self):
        """Print a detailed report of the test results."""
        m = self.metrics

        print("\n" + "=" * 70)
        print("GATEWAY STRESS TEST REPORT")
        print("=" * 70)

        if m.start_time and m.end_time:
            duration = m.end_time - m.start_time
            print(f"\nTest Duration: {duration:.1f} seconds")

        print(f"\n--- Request Summary ---")
        print(f"Total Requests: {m.total_requests}")
        print(f"Successful: {m.successful_requests}")
        print(f"Failed: {m.failed_requests}")
        print(f"Timeouts: {m.timeout_requests}")
        print(f"Success Rate: {m.success_rate:.2f}%")
        print(f"Throughput: {m.throughput:.2f} req/s")

        if m.latencies_ms:
            print(f"\n--- Latency Statistics (ms) ---")
            print(f"Min: {min(m.latencies_ms):.2f}")
            print(f"Max: {max(m.latencies_ms):.2f}")
            print(f"Avg: {m.avg_latency_ms:.2f}")
            print(f"P50: {m.p50_latency_ms:.2f}")
            print(f"P95: {m.p95_latency_ms:.2f}")
            print(f"P99: {m.p99_latency_ms:.2f}")
            if len(m.latencies_ms) > 1:
                print(f"Std Dev: {statistics.stdev(m.latencies_ms):.2f}")

        if m.errors_by_type:
            print(f"\n--- Errors by Type ---")
            for error_type, count in sorted(
                m.errors_by_type.items(), key=lambda x: -x[1]
            ):
                pct = count / m.total_requests * 100 if m.total_requests > 0 else 0
                print(f"  {error_type}: {count} ({pct:.1f}%)")

            if m.errors_messages:
                print(f"\n--- Sample Error Messages (first 5) ---")
                for msg in m.errors_messages[:5]:
                    print(f"  - {msg}")

        print("\n" + "=" * 70)

        # Summary verdict
        if m.success_rate >= 99:
            print("RESULT: EXCELLENT - Gateway performing well under load")
        elif m.success_rate >= 95:
            print("RESULT: GOOD - Minor issues detected")
        elif m.success_rate >= 90:
            print("RESULT: FAIR - Some reliability concerns")
        else:
            print("RESULT: POOR - Significant issues detected")

        print("=" * 70 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gateway Stress Tester for Marie-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic gRPC test (uses /api/v1/invoke and default API key)
    python gateway_stresser.py --gateway-port 52000

    # HTTP test (uses /api/v1/invoke and default API key)
    python gateway_stresser.py --protocol http --http-port 51000

    # High load test
    python gateway_stresser.py --protocol http --http-port 51000 --concurrency 50 --request-rate 100

    # Test specific endpoint (e.g., /extract)
    python gateway_stresser.py --protocol http --http-port 51000 --endpoint /extract

    # With custom request parameters (Marie gateway format)
    python gateway_stresser.py --protocol http --http-port 51000 \\
        --parameters '{"invoke_action": {"action_type": "command", "command": "job", "action": "submit", "name": "test"}}'

    # Target specific executor
    python gateway_stresser.py --protocol grpc --gateway-port 52000 --target-executor executor_a

    # Compare gRPC vs HTTP performance:
    python gateway_stresser.py --protocol grpc --gateway-port 52000 --duration 60
    python gateway_stresser.py --protocol http --http-port 51000 --duration 60

    # Without authentication (override default)
    python gateway_stresser.py --protocol http --http-port 51000 --api-key ""
        """,
    )

    parser.add_argument(
        "--gateway-host",
        default="localhost",
        help="Gateway host (default: localhost)",
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=52000,
        help="Gateway gRPC port (default: 52000)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=None,
        help="Gateway HTTP port (default: same as gateway-port)",
    )
    parser.add_argument(
        "--protocol",
        choices=["grpc", "http", "websocket"],
        default="grpc",
        help="Protocol to use (default: grpc)",
    )
    parser.add_argument(
        "--http-method",
        choices=["GET", "POST", "PUT"],
        default="POST",
        help="HTTP method to use (default: POST)",
    )
    parser.add_argument(
        "--endpoint",
        default="/api/v1/invoke",
        help="Endpoint to test (default: /api/v1/invoke)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=10.0,
        help="Target requests per second (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup period in seconds (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Documents per request (default: 1)",
    )
    parser.add_argument(
        "--document-text",
        default="Test document for stress testing",
        help="Text content for test documents",
    )
    parser.add_argument(
        "--no-direct-http",
        action="store_true",
        help="Use Marie Client for HTTP instead of direct aiohttp (default: use direct)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ",
        help="API key for authentication (Bearer token)",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        default=None,
        help="JSON string of request parameters (e.g., '{\"key\": \"value\"}')",
    )
    parser.add_argument(
        "--target-executor",
        type=str,
        default=None,
        help="Target executor name for the request",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine HTTP port
    http_port = args.http_port
    if http_port is None and args.protocol == "http":
        http_port = args.gateway_port

    # Parse JSON parameters if provided
    request_parameters = None
    if args.parameters:
        try:
            request_parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --parameters: {e}")
            sys.exit(1)

    stresser = GatewayStresser(
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        http_port=http_port,
        protocol=args.protocol,
        endpoint=args.endpoint,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        timeout=args.timeout,
        duration=args.duration,
        warmup=args.warmup,
        batch_size=args.batch_size,
        document_text=args.document_text,
        http_method=args.http_method,
        use_direct_http=not args.no_direct_http,
        api_key=args.api_key,
        request_parameters=request_parameters,
        target_executor=args.target_executor,
    )

    try:
        await stresser.run()
        stresser.print_report()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        stresser._running = False
        stresser.metrics.end_time = time.time()
        stresser.print_report()


if __name__ == "__main__":
    asyncio.run(main())
