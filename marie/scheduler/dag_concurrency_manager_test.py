import logging
import random
import time
from typing import Any, Dict
from unittest.mock import Mock

from marie.scheduler.dag_concurrency_manager import DagConcurrencyManager

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockScheduler:
    """Mock scheduler that simulates real scheduler behavior"""

    def __init__(self):
        self.base_slots = {
            'annotator_llm': 5,
            'annotator_embeddings': 1,
            'annotator_parser': 1,
            'annotator_table': 1,
            'annotator_table_parser': 0,
        }
        self.variation_enabled = False
        self.failure_mode = False

    def get_available_slots(self) -> Dict[str, int]:
        """Simulate getting available slots with optional variations"""
        if self.failure_mode:
            raise Exception("Scheduler connection failed")

        slots = self.base_slots.copy()

        if self.variation_enabled:
            # Add some realistic variations
            for slot_type in slots:
                variation = random.randint(-1, 2)
                slots[slot_type] = max(0, slots[slot_type] + variation)

        return slots

    def set_slots(self, **slots):
        """Update slot configuration"""
        self.base_slots.update(slots)

    def enable_variations(self, enabled=True):
        """Enable random variations in slot counts"""
        self.variation_enabled = enabled

    def set_failure_mode(self, enabled=True):
        """Simulate scheduler failures"""
        self.failure_mode = enabled


def example_1_basic_usage():
    """Example 1: Basic usage and capacity analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)

    # Initialize with mock scheduler
    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    # Get basic capacity calculation
    capacity = dag_manager.calculate_max_concurrent_dags()
    print(f"Calculated max concurrent DAGs: {capacity}")

    # Get detailed analysis
    analysis = dag_manager.get_capacity_analysis()
    print(f"\nDetailed Analysis:")
    print(f"Available slots: {analysis['available_slots']}")
    print(f"Bottleneck capacity: {analysis['bottleneck_capacity']}")
    print(f"Bottleneck resource: {analysis['bottleneck_resource']}")
    print(f"Final result: {analysis['bounded_capacity']}")

    # Show per-slot status
    print(f"\nSlot status:")
    for slot_type, count in analysis['available_slots'].items():
        status = (
            "⚠ BOTTLENECK"
            if count == analysis['bottleneck_capacity']
            else "✓ Available"
        )
        print(f"  {slot_type}: {count} slots {status}")


def example_2_limits_configuration():
    """Example 2: Configuring DAG limits"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: DAG Limits Configuration")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    print("Initial capacity:", dag_manager.calculate_max_concurrent_dags())

    # Update limits for a more conservative approach
    print("\n1. Setting conservative limits (min=2, max=10):")
    dag_manager.update_dag_limits(min_dags=2, max_dags=10)
    capacity_conservative = dag_manager.calculate_max_concurrent_dags()
    print(f"New capacity: {capacity_conservative}")

    # Set higher availability to test max limit
    print("\n2. Simulating high availability (all slots = 20):")
    scheduler.set_slots(
        annotator_llm=20,
        annotator_embeddings=20,
        annotator_parser=20,
        annotator_table=20,
        annotator_table_parser=20,
    )
    capacity_high = dag_manager.calculate_max_concurrent_dags()
    print(
        f"Capacity with high availability: {capacity_high} (should be capped at max=10)"
    )

    # Show configuration
    config = dag_manager.get_configuration_summary()
    print(config)
    print(
        f"\nCurrent limits: min={config['configuration']['dag_limits']['min']}, max={config['configuration']['dag_limits']['max']}"
    )


def example_3_utilization_factor():
    """Example 3: Utilization factor with pending jobs"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Utilization Factor with Pending Jobs")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    # Test with no pending jobs
    print("1. No pending jobs:")
    capacity_no_jobs = dag_manager.calculate_max_concurrent_dags()
    print(f"Capacity: {capacity_no_jobs}")

    # Test with light load
    print("\n2. Light load (2 pending jobs):")
    flat_jobs = [
        ('annotator_llm', {'job_id': 'job1'}),
        ('annotator_embeddings', {'job_id': 'job2'}),
    ]
    capacity_light = dag_manager.calculate_max_concurrent_dags(flat_jobs)
    analysis_light = dag_manager.get_capacity_analysis(flat_jobs)
    print(
        f"Capacity: {capacity_light} (utilization factor: {analysis_light['utilization_factor']:.3f})"
    )

    # Test with bottleneck (jobs waiting for unavailable slots)
    print("\n3. Bottleneck scenario (jobs waiting for unavailable slots):")
    flat_jobs_bottleneck = [
        ('annotator_table_parser', {'job_id': 'job1'}),  # 0 slots available
        ('annotator_table_parser', {'job_id': 'job2'}),  # 0 slots available
        ('annotator_llm', {'job_id': 'job3'}),
    ]
    capacity_bottleneck = dag_manager.calculate_max_concurrent_dags(
        flat_jobs_bottleneck
    )
    analysis_bottleneck = dag_manager.get_capacity_analysis(flat_jobs_bottleneck)
    print(
        f"Capacity: {capacity_bottleneck} (utilization factor: {analysis_bottleneck['utilization_factor']:.3f})"
    )

    # Test with severe bottleneck
    print("\n4. Severe bottleneck (many jobs waiting):")
    flat_jobs_severe = [
        ('annotator_table_parser', {'job_id': f'job{i}'}) for i in range(5)
    ]  # 5 jobs waiting for 0 slots
    capacity_severe = dag_manager.calculate_max_concurrent_dags(flat_jobs_severe)
    analysis_severe = dag_manager.get_capacity_analysis(flat_jobs_severe)
    print(
        f"Capacity: {capacity_severe} (utilization factor: {analysis_severe['utilization_factor']:.3f})"
    )


def example_4_capacity_tracking():
    """Example 4: Capacity tracking over time"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Capacity Tracking")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    # Initial state
    print("1. Initial capacity tracking:")
    tracking_info = dag_manager.get_capacity_tracking_info()
    print(f"Max observed capacity: {tracking_info['max_observed_capacity']}")

    # Simulate executor scaling up
    print("\n2. Simulating executor scale-up:")
    for step in range(3):
        # Gradually increase slots
        new_slots = {
            'annotator_llm': 5 + step * 2,
            'annotator_embeddings': 1 + step,
            'annotator_parser': 1 + step,
            'annotator_table': 1,
            'annotator_table_parser': step,  # Goes from 0 to 2
        }
        scheduler.set_slots(**new_slots)

        capacity = dag_manager.calculate_max_concurrent_dags()
        tracking_info = dag_manager.get_capacity_tracking_info()

        print(f"  Step {step + 1}: capacity={capacity}")
        print(f"    Current slots: {scheduler.get_available_slots()}")
        print(f"    Max observed: {tracking_info['max_observed_capacity']}")

    # Test with pending jobs after scale-up
    print("\n3. Testing with pending jobs after scale-up:")
    flat_jobs = [
        ('annotator_table_parser', {'job_id': f'job{i}'}) for i in range(2)
    ]  # Now we have capacity for these
    capacity_after_scale = dag_manager.calculate_max_concurrent_dags(flat_jobs)
    analysis_after_scale = dag_manager.get_capacity_analysis(flat_jobs)
    print(
        f"Capacity with jobs: {capacity_after_scale} (utilization factor: {analysis_after_scale['utilization_factor']:.3f})"
    )


def example_5_performance_and_caching():
    """Example 5: Performance testing and caching demonstration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Performance and Caching")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    # Test performance without cache
    print("Testing calculation performance:")
    start_time = time.time()
    for i in range(100):
        capacity = dag_manager.calculate_max_concurrent_dags()
    no_cache_time = time.time() - start_time
    print(f"100 calculations took: {no_cache_time:.3f} seconds")

    # Test with cache (same slots)
    start_time = time.time()
    for i in range(100):
        capacity = dag_manager.calculate_max_concurrent_dags()
    with_cache_time = time.time() - start_time
    print(f"100 cached calculations took: {with_cache_time:.3f} seconds")
    print(f"Cache speedup: {no_cache_time / with_cache_time:.1f}x")

    # Show metrics
    analysis = dag_manager.get_capacity_analysis()
    print(f"\nMetrics:")
    print(f"Total calculations: {analysis['calculation_count']}")
    print(f"Total errors: {analysis['error_count']}")
    print(f"Cache size: {len(dag_manager._cache)}")


def example_6_error_handling():
    """Example 6: Error handling and graceful degradation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Error Handling")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    # Normal operation
    print("1. Normal operation:")
    capacity = dag_manager.calculate_max_concurrent_dags()
    print(f"Capacity: {capacity}")

    # Simulate scheduler failure
    print("\n2. Scheduler failure scenario:")
    scheduler.set_failure_mode(True)
    capacity = dag_manager.calculate_max_concurrent_dags()  # Should return fallback
    print(f"Fallback capacity: {capacity}")

    # Restore scheduler and show recovery
    print("\n3. Recovery scenario:")
    scheduler.set_failure_mode(False)
    capacity = dag_manager.calculate_max_concurrent_dags()
    print(f"Recovered capacity: {capacity}")

    # Invalid configuration
    print("\n4. Invalid configuration handling:")
    try:
        dag_manager.update_dag_limits(min_dags=-1)  # Invalid min
    except Exception as e:
        print(f"Configuration error caught: {e}")

    try:
        dag_manager.update_dag_limits(max_dags=5, min_dags=10)  # max < min
    except Exception as e:
        print(f"Configuration error caught: {e}")

    # Health status
    health = dag_manager.get_health_status()
    print(f"\nHealth status: {health['status']}")
    if health['issues']:
        print(f"Issues: {health['issues']}")


def example_7_dynamic_slot_types():
    """Example 7: Handling dynamic slot types"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Dynamic Slot Types")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    print("Initial available slots:")
    initial_analysis = dag_manager.get_capacity_analysis()
    for slot_type, count in initial_analysis['available_slots'].items():
        print(f"  {slot_type}: {count}")

    # Add new slot type to scheduler
    print("\n1. Adding new slot type 'annotator_vision':")
    scheduler.set_slots(annotator_vision=3)

    capacity_with_new = dag_manager.calculate_max_concurrent_dags()
    new_analysis = dag_manager.get_capacity_analysis()
    print(f"Capacity with new slot: {capacity_with_new}")
    print(f"New bottleneck resource: {new_analysis['bottleneck_resource']}")

    # Show updated slots
    print("\n2. Updated available slots:")
    for slot_type, count in new_analysis['available_slots'].items():
        marker = "🆕" if slot_type == 'annotator_vision' else "  "
        bottleneck_marker = (
            " [BOTTLENECK]" if count == new_analysis['bottleneck_capacity'] else ""
        )
        print(f"{marker} {slot_type}: {count}{bottleneck_marker}")

    # Test capacity tracking with new slot type
    print("\n3. Capacity tracking for new slot type:")
    tracking_info = dag_manager.get_capacity_tracking_info()
    print(f"Max observed capacity: {tracking_info['max_observed_capacity']}")


def example_8_production_scheduler_integration():
    """Example 8: Integration with production scheduler"""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Production Scheduler Integration")
    print("=" * 60)

    # Example enhanced scheduler
    class EnhancedProductionScheduler:
        def __init__(self):
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            self.base_slots = {
                'annotator_llm': 8,
                'annotator_embeddings': 2,
                'annotator_parser': 3,
                'annotator_table': 2,
                'annotator_table_parser': 1,
                'annotator_vision': 1,  # New slot type
            }

            # Initialize concurrency manager
            self.dag_concurrency_manager = DagConcurrencyManager(
                self, min_concurrent_dags=2, max_concurrent_dags=25
            )

        def get_available_slots(self):
            """Your actual implementation - this is just a simulation"""
            return self.base_slots.copy()

        def submit_job(self, work_info, flat_jobs=None):
            """Enhanced job submission with adaptive concurrency"""
            try:
                # Calculate optimal concurrency
                self.max_concurrent_dags = (
                    self.dag_concurrency_manager.calculate_max_concurrent_dags(
                        flat_jobs
                    )
                )

                # Log decision
                analysis = self.dag_concurrency_manager.get_capacity_analysis(flat_jobs)
                self.logger.info(
                    f"Adaptive concurrency: {self.max_concurrent_dags} DAGs"
                )
                self.logger.debug(
                    f"Bottleneck resource: {analysis['bottleneck_resource']}"
                )

                # Simulate job submission
                print(
                    f"Submitting job with max_concurrent_dags: {self.max_concurrent_dags}"
                )
                return "job_id_12345"

            except Exception as e:
                self.logger.error(f"Error in adaptive submission: {e}")
                # Fallback to safe value
                self.max_concurrent_dags = 2
                self.logger.warning("Using fallback concurrency: 2 DAGs")
                return "job_id_fallback"

        def get_system_status(self):
            """Get comprehensive system status"""
            try:
                capacity_analysis = self.dag_concurrency_manager.get_capacity_analysis()
                health_status = self.dag_concurrency_manager.get_health_status()

                return {
                    'current_capacity': capacity_analysis['bounded_capacity'],
                    'slot_status': capacity_analysis['available_slots'],
                    'bottleneck': capacity_analysis['bottleneck_resource'],
                    'health': health_status,
                    'limits': capacity_analysis['limits'],
                }
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                return {'error': str(e), 'healthy': False}

    # Demonstrate usage
    scheduler = EnhancedProductionScheduler()

    # Submit a job with no pending work
    print("1. Job submission with no pending work:")
    job_id = scheduler.submit_job({'type': 'annotation_job', 'data': 'sample_data'})
    print(f"Job submitted: {job_id}")

    # Submit a job with some pending work
    print("\n2. Job submission with pending work:")
    flat_jobs = [
        ('annotator_table_parser', {'job_id': 'pending1'}),
        ('annotator_llm', {'job_id': 'pending2'}),
    ]
    job_id = scheduler.submit_job(
        {'type': 'annotation_job', 'data': 'sample_data'}, flat_jobs
    )
    print(f"Job submitted: {job_id}")

    # Get system status
    print("\n3. System status:")
    status = scheduler.get_system_status()
    print(f"Current capacity: {status['current_capacity']}")
    print(f"Bottleneck resource: {status['bottleneck']}")
    print(f"System healthy: {status['health']['status']}")
    print(f"Available slots: {status['slot_status']}")
    print(f"Configured limits: {status['limits']}")


def main():
    """Run all examples"""
    print("DAG Concurrency Manager - Simplified Sample Usage")
    print("=" * 60)

    try:
        example_1_basic_usage()
        example_2_limits_configuration()
        example_3_utilization_factor()
        example_4_capacity_tracking()
        example_5_performance_and_caching()
        example_6_error_handling()
        example_7_dynamic_slot_types()
        example_8_production_scheduler_integration()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
