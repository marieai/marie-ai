import logging

from marie.serve.discovery.etcd_client import EtcdClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import Mock, patch

from docarray import DocList
from docarray.documents import TextDoc

from marie.job.job_supervisor import JobSupervisor
from marie.types_core.request.data import DataRequest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JobSupervisorStresser:
    """
    A stress tester for the JobSupervisor's _send_callback_sync method.

    This class simulates a high volume of concurrent requests to test
    the performance and stability of etcd lease and put operations, especially
    with the LeaseCache optimization.
    """

    def __init__(self, use_lease_cache: bool = True):
        """
        :param use_lease_cache: If True, tests with LeaseCache. If False, tests the original, direct etcd.lease call path.
        """
        self.use_lease_cache = use_lease_cache
        self.supervisor = self._setup_supervisor()
        self.running = True
        logger.info(
            f"Stresser initialized. Lease cache enabled: {self.use_lease_cache}"
        )

    def _setup_supervisor(self) -> JobSupervisor:
        """Sets up the JobSupervisor with mocked dependencies."""
        etcd_client = Mock()
        etcd_client = EtcdClient("localhost", 2379)
        lease_cache_mock = None

        # To test both scenarios (with and without cache), we conditionally patch
        # the supervisor's _send_callback_sync to use either the cache or direct etcd calls.
        if False:
            if self.use_lease_cache:
                # When testing with the cache, we mock the LeaseCache itself
                with patch(
                    'marie.job.job_supervisor.LeaseCache'
                ) as mock_lease_cache_class:
                    lease_cache_mock = mock_lease_cache_class.return_value
                    lease_cache_mock.get_or_refresh.return_value = Mock()
            else:
                # When testing without the cache, we mock the direct etcd client lease method
                etcd_client.lease.return_value = Mock()

        supervisor = JobSupervisor(
            job_id="stress-test-job",
            job_info_client=Mock(),
            job_distributor=Mock(),
            event_publisher=Mock(),
            etcd_client=etcd_client,
            confirmation_event=asyncio.Event(),
        )

        # Attach the mocked lease cache if it's being used
        if lease_cache_mock:
            supervisor._lease_cache = lease_cache_mock

        # Replace the logger with our configured logger
        supervisor.logger = logger

        return supervisor

    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parses a duration string like '10h', '30m', '5s' into a timedelta object."""
        if not duration_str:
            return timedelta()

        # Simple regex to capture value and unit
        match = re.match(r"(\d+)\s*([hms])?$", duration_str.lower().strip())
        if not match:
            raise ValueError(
                f"Invalid duration format: '{duration_str}'. Use '10h', '30m', '300s'."
            )

        value = int(match.group(1))
        unit = match.group(2) or 's'  # Default to seconds if no unit

        if unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        else:  # 's'
            return timedelta(seconds=value)

    def _send_one_request(self, request_idx: int):
        """Simulates sending a single callback with randomized data."""
        try:
            # Simulate a variety of addresses and deployments to test cache effectiveness
            address = (
                f"172.20.10.{10 + (request_idx % 25)}:{50000 + (request_idx % 2000)}"
            )
            deployment = f"executor-deployment-{(request_idx % 50)}"

            request_info = {
                "request_id": f"stress-{uuid.uuid4()}",
                "address": address,
                "deployment": deployment,
            }
            request = DataRequest()
            request.data.docs = DocList[TextDoc](
                [TextDoc(text=f"payload-{request_idx}")]
            )

            # Call the method under test
            self.supervisor._send_callback_sync([request], request_info)
            return True
        except Exception as e:
            logger.error(f"Error processing request {request_idx}: {e}", exc_info=False)
            return False

    async def run_stress_test(
        self, num_requests: int = 0, run_time: str = "", concurrency: int = 50
    ):
        """
        Runs a high-volume stress test for a specified number of requests or a set duration.

        :param num_requests: Total number of requests to simulate. The test stops when this many requests are processed.
        :param run_time: The duration to run the test for (e.g., "10s", "5m", "1h"). The test stops after this duration.
        :param concurrency: Number of parallel threads sending requests.
        """
        duration = self._parse_duration(run_time)
        end_time = (
            time.monotonic() + duration.total_seconds()
            if duration.total_seconds() > 0
            else None
        )

        if num_requests <= 0 and not end_time:
            raise ValueError(
                "Either `num_requests` (must be > 0) or `run_time` must be specified."
            )

        log_msg = "Starting stress test:"
        if num_requests > 0:
            log_msg += f" {num_requests} requests"
        if end_time:
            if num_requests > 0:
                log_msg += " or"
            log_msg += f" for {run_time}"
        log_msg += f" with {concurrency} workers."
        logger.info(log_msg)

        success_count = 0
        failure_count = 0
        start_time = time.time()

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = set()
            request_index = 0

            while self.running:
                # Primary exit condition check
                if end_time and time.monotonic() >= end_time:
                    logger.info(
                        "Run time limit reached. Waiting for outstanding tasks to complete..."
                    )
                    break
                if num_requests > 0 and (success_count + failure_count) >= num_requests:
                    logger.info(
                        "Request limit reached. Waiting for outstanding tasks to complete..."
                    )
                    break

                # Refill the futures set up to the concurrency limit
                while len(futures) < concurrency and (
                    num_requests <= 0 or request_index < num_requests
                ):
                    # Check for time limit again before submitting a new task
                    if end_time and time.monotonic() >= end_time:
                        break

                    future = loop.run_in_executor(
                        executor, self._send_one_request, request_index
                    )
                    futures.add(future)
                    request_index += 1

                if not futures:
                    # This happens if all requests are submitted and completed
                    break

                # Wait for at least one future to complete
                done, futures = await asyncio.wait(
                    futures, return_when=asyncio.FIRST_COMPLETED
                )

                for future in done:
                    try:
                        if future.result():
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        logger.error(
                            f"Request future failed with an exception: {e}",
                            exc_info=False,
                        )
                        failure_count += 1

                processed_count = success_count + failure_count
                # Log progress periodically
                if processed_count > 0 and processed_count % (concurrency * 5) == 0:
                    logger.info(f"Progress: {processed_count} requests processed...")

        # After loop, cancel any futures that are still running (if we timed out)
        if futures:
            logger.info(f"Cancelling {len(futures)} outstanding in-flight tasks.")
            for f in futures:
                f.cancel()
            failure_count += len(futures)  # Consider cancelled tasks as failed

        end_time_run = time.time()
        duration_run = end_time_run - start_time
        total_processed = success_count + failure_count
        rps = total_processed / duration_run if duration_run > 0 else float('inf')

        logger.info("Stress test finished.")
        logger.info("--- Stress Test Summary ---")
        logger.info(f"Total requests processed: {total_processed:,}")
        logger.info(f"Successful: {success_count:,}, Failed: {failure_count:,}")
        logger.info(f"Total duration: {duration_run:.2f} seconds")
        logger.info(f"Throughput: {rps:,.2f} requests/sec")
        logger.info("--------------------------")

    def stop(self):
        """Stops the stress test gracefully."""
        logger.info("Stopping stress test...")
        self.running = False


if __name__ == "__main__":
    # This allows you to run the script directly from the command line.
    # You can adjust the numbers to simulate millions of requests.
    stresser = None

    async def main():
        global stresser
        stresser = JobSupervisorStresser(use_lease_cache=False)
        # Example 1: Run a time-based test for 15 seconds.
        logger.info("\n--- Running time-based stress test (15 seconds) ---")
        await stresser.run_stress_test(run_time="30s", concurrency=10)
        #
        # # Example 2: Run a request-based test for 20,000 requests.
        # logger.info("\n--- Running request-based stress test (20,000 requests) ---")
        # await stresser.run_stress_test(num_requests=20_000, concurrency=10)

        # Example 3: Run for a long duration, e.g., 1 hour.
        # logger.info("\n--- Starting 1-hour soak test ---")
        # await stresser.run_stress_test(run_time="1h", concurrency=100)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stress test interrupted by user.")
    finally:
        if stresser:
            stresser.stop()

# python -m tools.stress.job_supervisor_stresser > stress_test.log 2>&1
# python -m tools.stress.parse_stress_log stress_test.log
