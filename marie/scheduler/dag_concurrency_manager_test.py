"""
DAG Concurrency Manager - Sample Usage
"""

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
    print(f"Weighted capacity: {analysis['calculations']['weighted_average']['value']}")
    print(f"Conservative capacity: {analysis['calculations']['conservative']['value']}")
    print(f"Final result: {analysis['final_result']}")

    # Show per-slot breakdown
    print(f"\nPer-slot breakdown:")
    for slot_type, breakdown in analysis['slot_breakdown'].items():
        defaults = []
        if breakdown['using_defaults']['multiplier']:
            defaults.append('mult')
        if breakdown['using_defaults']['weight']:
            defaults.append('weight')
        defaults_str = f" (defaults: {','.join(defaults)})" if defaults else ""
        print(
            f"  {slot_type}: {breakdown['slots']} slots × {breakdown['multiplier']} mult × {breakdown['weight']:.3f} weight = {breakdown['weighted_contribution']:.2f}{defaults_str}"
        )


def example_2_configuration_updates():
    """Example 2: Dynamic configuration updates"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Configuration Updates")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    print("Initial capacity:", dag_manager.calculate_max_concurrent_dags())

    # Update strategy balance to be more conservative
    print("\n1. Making strategy more conservative (30% weighted, 70% conservative):")
    dag_manager.update_strategy_balance(0.3, 0.7)
    capacity_conservative = dag_manager.calculate_max_concurrent_dags()
    print(f"New capacity: {capacity_conservative}")

    # Update resource weights to prioritize LLM slots
    print("\n2. Prioritizing LLM slots (80% weight):")
    dag_manager.update_resource_weights(
        annotator_llm=0.8,
        annotator_embeddings=0.1,
        annotator_parser=0.05,
        annotator_table=0.03,
        annotator_table_parser=0.02,
    )
    capacity_llm_focused = dag_manager.calculate_max_concurrent_dags()
    print(f"New capacity: {capacity_llm_focused}")

    # Update slot multipliers
    print("\n3. Increasing LLM multiplier (can handle more DAGs per slot):")
    dag_manager.update_slot_multiplier('annotator_llm', 4)
    capacity_higher_mult = dag_manager.calculate_max_concurrent_dags()
    print(f"New capacity: {capacity_higher_mult}")

    # Show final configuration
    analysis = dag_manager.get_capacity_analysis()
    print(
        f"\nFinal configuration multipliers: {[(k, analysis['slot_breakdown'][k]['multiplier']) for k in analysis['slot_breakdown']]}"
    )
    print(
        f"Final configuration weights: {[(k, analysis['slot_breakdown'][k]['weight']) for k in analysis['slot_breakdown']]}"
    )


def example_3_performance_and_caching():
    """Example 3: Performance testing and caching demonstration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Performance and Caching")
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
    metrics = analysis['metrics']
    print(f"\nMetrics:")
    print(f"Total calculations: {metrics['calculation_count']}")
    print(f"Total errors: {metrics['error_count']}")
    print(f"Cache size: {metrics['cache_size']}")
    if metrics['last_calculation']:
        print(
            f"Last calculation time: {metrics['last_calculation']['calculation_time_ms']:.2f} ms"
        )


def example_4_error_handling():
    """Example 4: Error handling and graceful degradation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Error Handling")
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
    try:
        capacity = dag_manager.calculate_max_concurrent_dags()
        print(f"Fallback capacity: {capacity}")
    except Exception as e:
        print(f"Error caught: {e}")

    # Restore scheduler and show recovery
    print("\n3. Recovery scenario:")
    scheduler.set_failure_mode(False)
    capacity = dag_manager.calculate_max_concurrent_dags()
    print(f"Recovered capacity: {capacity}")

    # Invalid configuration
    print("\n4. Invalid configuration handling:")
    try:
        dag_manager.update_slot_multiplier('test_slot', -1)  # Invalid multiplier
    except Exception as e:
        print(f"Configuration error caught: {e}")

    # Health status
    health = dag_manager.get_health_status()
    print(f"\nHealth status: {health}")


def example_5_dynamic_slot_types():
    """Example 5: Handling dynamic slot types"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Dynamic Slot Types")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    print("Initial slot types:")
    status = dag_manager.get_slot_types_status()
    for slot_type, info in status.items():
        print(f"  {slot_type}: configured={info['fully_configured']}")

    # Add new slot type to scheduler
    print("\n1. Adding new slot type 'annotator_vision' to scheduler:")
    scheduler.set_slots(annotator_vision=3)

    capacity_before = dag_manager.calculate_max_concurrent_dags()
    print(f"Capacity with new slot (using defaults): {capacity_before}")

    # Configure the new slot type
    print("\n2. Configuring new slot type:")
    dag_manager.add_slot_type_config('annotator_vision', multiplier=2, weight=0.15)

    capacity_after = dag_manager.calculate_max_concurrent_dags()
    print(f"Capacity with configured slot: {capacity_after}")

    # Show updated status
    print("\n3. Updated slot types status:")
    status = dag_manager.get_slot_types_status()
    for slot_type, info in status.items():
        configured = "✓" if info['fully_configured'] else "⚠"
        print(
            f"  {configured} {slot_type}: slots={info['available_slots']}, mult={info['multiplier']}, weight={info['weight']:.3f}"
        )


def example_6_monitoring_setup():
    """Example 6: Production monitoring setup"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Production Monitoring")
    print("=" * 60)

    scheduler = MockScheduler()
    dag_manager = DagConcurrencyManager(scheduler)

    # Enable slot variations to simulate real environment
    scheduler.enable_variations(True)

    print("Simulating production workload...")

    # Collect metrics over time
    metrics_history = []
    for i in range(10):
        start_time = time.time()
        capacity = dag_manager.calculate_max_concurrent_dags()

        # Get health status
        health = dag_manager.get_health_status()
        analysis = dag_manager.get_capacity_analysis()

        metrics = {
            'timestamp': start_time,
            'capacity': capacity,
            'healthy': health['healthy'],
            'calculation_time_ms': (
                analysis['metrics']['last_calculation']['calculation_time_ms']
                if analysis['metrics']['last_calculation']
                else 0
            ),
            'bottleneck': (
                analysis['metrics']['last_calculation']['bottleneck_resource']
                if analysis['metrics']['last_calculation']
                else None
            ),
            'available_slots': dict(analysis['available_slots']),
        }
        metrics_history.append(metrics)

        print(
            f"Step {i + 1}: capacity={capacity}, time={metrics['calculation_time_ms']:.2f}ms, bottleneck={metrics['bottleneck']}"
        )

        # Small delay to simulate real timing
        time.sleep(0.1)

    # Analysis
    print(f"\nMonitoring Summary:")
    avg_capacity = sum(m['capacity'] for m in metrics_history) / len(metrics_history)
    avg_time = sum(m['calculation_time_ms'] for m in metrics_history) / len(
        metrics_history
    )
    bottlenecks = [m['bottleneck'] for m in metrics_history if m['bottleneck']]
    most_common_bottleneck = (
        max(set(bottlenecks), key=bottlenecks.count) if bottlenecks else "None"
    )

    print(f"Average capacity: {avg_capacity:.1f}")
    print(f"Average calculation time: {avg_time:.2f} ms")
    print(f"Most common bottleneck: {most_common_bottleneck}")
    print(f"All calculations healthy: {all(m['healthy'] for m in metrics_history)}")

    # Final health check
    final_health = dag_manager.get_health_status()
    print(f"\nFinal health status:")
    for key, value in final_health.items():
        print(f"  {key}: {value}")


def example_7_production_scheduler_integration():
    """Example 7: Integration with production scheduler"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Production Scheduler Integration")
    print("=" * 60)

    # This would be your actual scheduler class
    class ProductionScheduler:
        def __init__(self):
            # Your actual initialization
            pass

        def get_available_slots(self):
            # Your actual implementation
            # This is just a simulation
            return {
                'annotator_llm': 8,
                'annotator_embeddings': 2,
                'annotator_parser': 3,
                'annotator_table': 2,
                'annotator_table_parser': 1,
                'annotator_vision': 1,  # New slot type
            }

        def submit_job(self, work_info, overwrite=False):
            # Your actual job submission logic
            print(
                f"Submitting job with max_concurrent_dags: {self.max_concurrent_dags}"
            )
            return "job_id_12345"

    # Example enhanced scheduler
    class EnhancedProductionScheduler(ProductionScheduler):
        def __init__(self):
            super().__init__()
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

            # Initialize concurrency manager
            self.dag_concurrency_manager = DagConcurrencyManager(self)

            # Set custom configuration for your environment
            self.dag_concurrency_manager.update_resource_weights(
                annotator_llm=0.35,  # High priority
                annotator_embeddings=0.25,  # Medium-high priority
                annotator_parser=0.20,  # Medium priority
                annotator_table=0.10,  # Lower priority
                annotator_table_parser=0.05,  # Low priority
                annotator_vision=0.05,  # New feature, low priority initially
            )

            # Conservative strategy for production stability
            self.dag_concurrency_manager.update_strategy_balance(
                0.4, 0.6
            )  # 40% weighted, 60% conservative

            # Production limits
            self.dag_concurrency_manager.update_dag_limits(
                min_concurrent_dags=2, max_concurrent_dags=25
            )

        def submit_job(self, work_info, overwrite=False):
            """Enhanced job submission with adaptive concurrency"""
            try:
                # Calculate optimal concurrency
                self.max_concurrent_dags = (
                    self.dag_concurrency_manager.calculate_max_concurrent_dags()
                )

                # Log decision
                analysis = self.dag_concurrency_manager.get_capacity_analysis()
                self.logger.info(
                    f"Adaptive concurrency: {self.max_concurrent_dags} DAGs"
                )
                self.logger.debug(
                    f"Bottleneck resource: {analysis['metrics']['last_calculation']['bottleneck_resource'] if analysis['metrics']['last_calculation'] else 'Unknown'}"
                )

                # Submit with calculated concurrency
                return super().submit_job(work_info, overwrite)

            except Exception as e:
                self.logger.error(f"Error in adaptive submission: {e}")
                # Fallback to safe value
                self.max_concurrent_dags = 2
                self.logger.warning("Using fallback concurrency: 2 DAGs")
                return super().submit_job(work_info, overwrite)

        def get_system_status(self):
            """Get comprehensive system status"""
            try:
                capacity_analysis = self.dag_concurrency_manager.get_capacity_analysis()
                health_status = self.dag_concurrency_manager.get_health_status()

                return {
                    'current_capacity': capacity_analysis['final_result'],
                    'slot_status': capacity_analysis['available_slots'],
                    'bottleneck': (
                        capacity_analysis['metrics']['last_calculation'][
                            'bottleneck_resource'
                        ]
                        if capacity_analysis['metrics']['last_calculation']
                        else None
                    ),
                    'health': health_status,
                    'configuration': {
                        'strategy_weights': {
                            'weighted': self.dag_concurrency_manager.weighted_strategy_weight,
                            'conservative': self.dag_concurrency_manager.conservative_strategy_weight,
                        },
                        'limits': {
                            'min': self.dag_concurrency_manager.min_concurrent_dags,
                            'max': self.dag_concurrency_manager.max_concurrent_dags,
                        },
                    },
                }
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                return {'error': str(e), 'healthy': False}

    # Demonstrate usage
    scheduler = EnhancedProductionScheduler()

    # Submit a job
    job_id = scheduler.submit_job({'type': 'annotation_job', 'data': 'sample_data'})
    print(f"Job submitted: {job_id}")

    # Get system status
    status = scheduler.get_system_status()
    print(f"\nSystem Status:")
    print(f"Current capacity: {status['current_capacity']}")
    print(f"Bottleneck resource: {status['bottleneck']}")
    print(f"System healthy: {status['health']['healthy']}")
    print(f"Available slots: {status['slot_status']}")


def main():
    """Run all examples"""
    print("DAG Concurrency Manager - Comprehensive Sample Usage")
    print("=" * 60)

    try:
        example_1_basic_usage()
        example_2_configuration_updates()
        example_3_performance_and_caching()
        example_4_error_handling()
        example_5_dynamic_slot_types()
        example_6_monitoring_setup()
        example_7_production_scheduler_integration()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
