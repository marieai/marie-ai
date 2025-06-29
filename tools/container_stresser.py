#!/usr/bin/env python3
"""
Docker Container Stresser

A script to start and stop Docker containers at random intervals or with specific patterns
to test application resilience and reconnection behavior.
"""

import argparse
import logging
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('container_stresser.log')],
)
logger = logging.getLogger(__name__)


class DockerContainerStresser:
    """Docker container stress testing utility."""

    def __init__(self, container_name):
        self.container_name = container_name
        self.running = True
        self.stats = {
            'restarts': 0,
            'stops': 0,
            'starts': 0,
            'failures': 0,
            'start_time': datetime.now(),
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping stress test...")
        self.running = False

    def _run_docker_command(self, command, timeout=30):
        """Run a docker command and return success status."""
        try:
            cmd = ["docker"] + command + [self.container_name]
            logger.debug(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0:
                logger.info(f"Command '{' '.join(command)}' successful")
                return True
            else:
                logger.error(f"Command failed: {result.stderr.strip()}")
                self.stats['failures'] += 1
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Command '{' '.join(command)}' timed out")
            self.stats['failures'] += 1
            return False
        except Exception as e:
            logger.error(f"Error running command: {e}")
            self.stats['failures'] += 1
            return False

    def get_container_status(self):
        """Get current container status."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    self.container_name,
                    "--format",
                    "{{.State.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "not_found"
        except:
            return "error"

    def start_container(self):
        """Start the container."""
        logger.info(f"Starting container: {self.container_name}")
        if self._run_docker_command(["start"]):
            self.stats['starts'] += 1
            return True
        return False

    def stop_container(self):
        """Stop the container."""
        logger.info(f"Stopping container: {self.container_name}")
        if self._run_docker_command(["stop"]):
            self.stats['stops'] += 1
            return True
        return False

    def restart_container(self):
        """Restart the container."""
        logger.info(f"Restarting container: {self.container_name}")
        if self._run_docker_command(["restart"]):
            self.stats['restarts'] += 1
            return True
        return False

    def kill_container(self):
        """Kill the container (hard stop)."""
        logger.info(f"Killing container: {self.container_name}")
        if self._run_docker_command(["kill"]):
            self.stats['stops'] += 1
            return True
        return False

    def wait_for_container_ready(self, timeout=60):
        """Wait for container to be running."""
        start_time = time.time()
        while time.time() - start_time < timeout and self.running:
            status = self.get_container_status()
            if status == "running":
                return True
            time.sleep(1)
        return False

    def print_stats(self):
        """Print current statistics."""
        duration = datetime.now() - self.stats['start_time']
        print(f"\n{'=' * 50}")
        print(f"Container Stress Test Statistics")
        print(f"{'=' * 50}")
        print(f"Container: {self.container_name}")
        print(f"Duration: {duration}")
        print(f"Restarts: {self.stats['restarts']}")
        print(f"Stops: {self.stats['stops']}")
        print(f"Starts: {self.stats['starts']}")
        print(f"Failures: {self.stats['failures']}")
        print(f"Current Status: {self.get_container_status()}")
        print(f"{'=' * 50}\n")

    def stress_test_random(self, duration_minutes=10, min_interval=5, max_interval=30):
        """Random restart stress test."""
        logger.info(f"Starting random stress test for {duration_minutes} minutes")
        logger.info(f"Random interval: {min_interval}-{max_interval} seconds")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time and self.running:
            # Random action
            action = random.choice(['restart', 'stop_start', 'kill_start'])

            if action == 'restart':
                self.restart_container()
            elif action == 'stop_start':
                self.stop_container()
                time.sleep(random.uniform(1, 5))  # Brief downtime
                self.start_container()
            elif action == 'kill_start':
                self.kill_container()
                time.sleep(random.uniform(1, 3))  # Brief downtime
                self.start_container()

            # Wait for container to be ready
            self.wait_for_container_ready()

            # Random interval before next action
            if self.running:
                interval = random.uniform(min_interval, max_interval)
                logger.info(f"Waiting {interval:.1f} seconds before next action...")
                time.sleep(interval)

        logger.info("Random stress test completed")

    def stress_test_periodic(self, count=10, interval=15, action='restart'):
        """Periodic stress test with fixed intervals."""
        logger.info(
            f"Starting periodic stress test: {count} {action}s every {interval} seconds"
        )

        for i in range(count):
            if not self.running:
                break

            logger.info(f"Action {i + 1}/{count}")

            if action == 'restart':
                self.restart_container()
            elif action == 'stop':
                self.stop_container()
                time.sleep(2)
                self.start_container()
            elif action == 'kill':
                self.kill_container()
                time.sleep(2)
                self.start_container()

            # Wait for container to be ready
            self.wait_for_container_ready()

            # Wait for next iteration
            if i < count - 1 and self.running:
                logger.info(f"Waiting {interval} seconds...")
                time.sleep(interval)

        logger.info("Periodic stress test completed")

    def stress_test_burst(self, burst_count=5, burst_interval=2, rest_period=30):
        """Burst stress test - rapid restarts followed by rest periods."""
        logger.info(
            f"Starting burst stress test: {burst_count} restarts every {burst_interval}s, then {rest_period}s rest"
        )

        burst_number = 1
        while self.running:
            logger.info(f"Starting burst {burst_number}")

            # Perform burst of restarts
            for i in range(burst_count):
                if not self.running:
                    break

                logger.info(f"Burst restart {i + 1}/{burst_count}")
                self.restart_container()

                if i < burst_count - 1:  # Don't wait after last restart in burst
                    time.sleep(burst_interval)

            # Rest period
            if self.running:
                logger.info(f"Resting for {rest_period} seconds...")
                time.sleep(rest_period)
                burst_number += 1

        logger.info("Burst stress test completed")

    def stress_test_chaos(self, duration_minutes=15):
        """Chaos test - completely random timing and actions."""
        logger.info(f"Starting chaos stress test for {duration_minutes} minutes")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        actions = ['restart', 'stop', 'kill', 'start']

        while datetime.now() < end_time and self.running:
            action = random.choice(actions)

            # Get current status to make intelligent decisions
            status = self.get_container_status()

            if status == "running":
                if action in ['restart', 'stop', 'kill']:
                    if action == 'restart':
                        self.restart_container()
                    elif action == 'stop':
                        self.stop_container()
                    elif action == 'kill':
                        self.kill_container()
            elif status in ["exited", "stopped"]:
                if action == 'start':
                    self.start_container()
                else:
                    # Force a start if container is down
                    self.start_container()

            # Random wait time
            wait_time = random.uniform(1, 20)
            logger.info(f"Chaos wait: {wait_time:.1f} seconds")
            time.sleep(wait_time)

        logger.info("Chaos stress test completed")


def main():
    parser = argparse.ArgumentParser(description="Docker Container Stress Tester")
    parser.add_argument("--container", required=True, help="Docker container name")
    parser.add_argument(
        "--mode",
        choices=['random', 'periodic', 'burst', 'chaos'],
        default='random',
        help="Stress test mode",
    )

    # Random mode options
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in minutes (default: 10)",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=5,
        help="Minimum interval between actions in seconds (default: 5)",
    )
    parser.add_argument(
        "--max-interval",
        type=int,
        default=30,
        help="Maximum interval between actions in seconds (default: 30)",
    )

    # Periodic mode options
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of actions for periodic mode (default: 10)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Interval between actions in seconds (default: 15)",
    )
    parser.add_argument(
        "--action",
        choices=['restart', 'stop', 'kill'],
        default='restart',
        help="Action type for periodic mode",
    )

    # Burst mode options
    parser.add_argument(
        "--burst-count",
        type=int,
        default=5,
        help="Number of restarts per burst (default: 5)",
    )
    parser.add_argument(
        "--burst-interval",
        type=int,
        default=2,
        help="Interval between restarts in burst (default: 2)",
    )
    parser.add_argument(
        "--rest-period",
        type=int,
        default=30,
        help="Rest period between bursts (default: 30)",
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create stresser
    stresser = DockerContainerStresser(args.container)

    # Check if container exists
    status = stresser.get_container_status()
    if status == "not_found":
        logger.error(f"Container '{args.container}' not found")
        sys.exit(1)

    logger.info(f"Container '{args.container}' found with status: {status}")

    try:
        # Run stress test based on mode
        if args.mode == 'random':
            stresser.stress_test_random(
                duration_minutes=args.duration,
                min_interval=args.min_interval,
                max_interval=args.max_interval,
            )
        elif args.mode == 'periodic':
            stresser.stress_test_periodic(
                count=args.count, interval=args.interval, action=args.action
            )
        elif args.mode == 'burst':
            stresser.stress_test_burst(
                burst_count=args.burst_count,
                burst_interval=args.burst_interval,
                rest_period=args.rest_period,
            )
        elif args.mode == 'chaos':
            stresser.stress_test_chaos(duration_minutes=args.duration)

    except KeyboardInterrupt:
        logger.info("Stress test interrupted by user")
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        sys.exit(1)
    finally:
        # Print final statistics
        stresser.print_stats()

        # Ensure container is running before exit
        if stresser.get_container_status() != "running":
            logger.info("Starting container before exit...")
            stresser.start_container()


if __name__ == "__main__":
    main()
