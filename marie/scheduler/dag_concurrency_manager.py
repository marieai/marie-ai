import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from marie.excepts import BadConfigSource
from marie.scheduler.job_scheduler import JobScheduler


class CapacityCalculationError(Exception):
    """Exception raised when capacity calculation fails"""

    pass


class CapacityMetrics:
    """Metrics for capacity calculation performance and results"""

    def __init__(
        self,
        calculation_time_ms: float,
        bottleneck_capacity: int,
        final_capacity: int,
        bottleneck_resource: Optional[str] = None,
        utilization_factor: Optional[float] = None,
        timestamp: int = None,
    ):
        self.calculation_time_ms = calculation_time_ms
        self.bottleneck_capacity = bottleneck_capacity
        self.final_capacity = final_capacity
        self.bottleneck_resource = bottleneck_resource
        self.utilization_factor = utilization_factor
        # Use integer milliseconds timestamp
        self.timestamp = timestamp or self._get_current_timestamp_ms()

    def _get_current_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds as integer"""
        return int(time.time() * 1000)

    def to_export_dict(self, prefix: str = 'dag_concurrency_') -> Dict:
        """Export metrics in monitoring-friendly format"""
        current_time_ms = self._get_current_timestamp_ms()

        metrics = {
            f'{prefix}current_capacity': self.final_capacity,
            f'{prefix}bottleneck_capacity': self.bottleneck_capacity,
            f'{prefix}bottleneck_resource': self.bottleneck_resource or 'none',
            f'{prefix}last_calculation_timestamp': self.timestamp,
            f'{prefix}calculation_age_seconds': max(
                0, (current_time_ms - self.timestamp) / 1000.0
            ),
        }

        # Add utilization factor if available
        if self.utilization_factor is not None:
            metrics.update(
                {
                    f'{prefix}utilization_factor': self.utilization_factor,
                    f'{prefix}has_pending_jobs': 1,
                }
            )
        else:
            metrics[f'{prefix}has_pending_jobs'] = 0

        return metrics


class DagConcurrencyManager:
    """
    Manages workload execution capacity based on available system resources.

    Calculates optimal concurrent execution limits by analyzing:
    - Available resource slots
    - Current workload demand
    - Historical capacity patterns
    - Resource bottlenecks and utilization
    """

    def __init__(
        self,
        scheduler: JobScheduler,
        strategy: str = 'dynamic',
        min_concurrent_dags: int = 1,
        max_concurrent_dags: int = 16,
        cache_ttl_seconds: int = 10,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Validate parameters
        if min_concurrent_dags < 0:
            raise BadConfigSource(
                f"min_concurrent_dags must be >= 0: {min_concurrent_dags}"
            )
        if max_concurrent_dags < min_concurrent_dags:
            raise BadConfigSource(
                f"max_concurrent_dags ({max_concurrent_dags}) < min_concurrent_dags ({min_concurrent_dags})"
            )

        _ALLOWED_STRATEGIES = {'fixed', 'dynamic'}
        if strategy not in _ALLOWED_STRATEGIES:
            raise BadConfigSource(
                f"Invalid strategy '{strategy}'. Must be one of: {', '.join(_ALLOWED_STRATEGIES)}"
            )
        self.strategy = strategy

        self.min_concurrent_dags = min_concurrent_dags
        self.max_concurrent_dags = max_concurrent_dags
        self._cache_ttl_seconds = cache_ttl_seconds

        self.scheduler = scheduler
        self._lock = threading.RLock()

        self.last_calculation_metrics: Optional[CapacityMetrics] = None
        self._calculation_count = 0
        self._error_count = 0

        self._cache = {}
        self._max_observed_capacity = {}

    def _get_current_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds as integer"""
        return int(time.time() * 1000)

    def _get_cache_key(self, flat_jobs: List = None) -> str:
        """Generate cache key based on current state"""
        try:
            available_slots = self.scheduler.get_available_slots()
            slots_key = hash(tuple(sorted(available_slots.items())))

            # Include flat_jobs in cache key if provided
            if flat_jobs is not None:
                jobs_by_slot = {}
                for slot_name, work_info in flat_jobs:
                    jobs_by_slot[slot_name] = jobs_by_slot.get(slot_name, 0) + 1
                jobs_key = hash(tuple(sorted(jobs_by_slot.items())))
                return f"{slots_key}_{jobs_key}"
            return str(slots_key)
        except Exception:
            return f"error_{self._get_current_timestamp_ms()}"

    def _is_cache_valid(self, cache_entry) -> bool:
        """Check if cache entry is still valid"""
        current_time_ms = self._get_current_timestamp_ms()
        age_seconds = (current_time_ms - cache_entry['timestamp']) / 1000.0
        return age_seconds < self._cache_ttl_seconds

    def _update_capacity_tracking(self, available_slots: Dict[str, int]):
        """
        Track the maximum observed capacity for each slot type.
        This helps us understand the normal operating capacity.
        """
        with self._lock:
            for slot_type, current_count in available_slots.items():
                current_max = self._max_observed_capacity.get(slot_type, 0)
                if current_count > current_max:
                    self._max_observed_capacity[slot_type] = current_count
                    self.logger.debug(
                        f"Updated max observed capacity for {slot_type}: {current_count}"
                    )

    def _get_expected_capacity(self, slot_type: str) -> int:
        """
        Get the expected normal capacity for a slot type.
        Uses max observed capacity as the baseline.
        """
        max_observed = self._max_observed_capacity.get(slot_type, 0)

        # If we haven't seen this slot type yet, use 1 as estimate
        if max_observed == 0:
            return 1

        return max_observed

    def _find_bottleneck_resource(
        self, available_slots: Dict[str, int]
    ) -> Optional[str]:
        """Find the most constraining resource"""
        if not available_slots:
            return None

        min_slots = min(available_slots.values())
        for slot_type, slots in available_slots.items():
            if slots == min_slots:
                return slot_type
        return None

    def calculate_max_concurrent_dags(
        self, flat_jobs: List[Tuple[str, Any]] = None
    ) -> int:
        """
        Simplified approach: Conservative bottleneck calculation with utilization adjustment
        """

        if self.strategy == 'fixed':
            return self.min_concurrent_dags

        with self._lock:
            start_time = time.time()

            try:
                # Check cache first
                cache_key = self._get_cache_key(flat_jobs)
                if cache_key in self._cache and self._is_cache_valid(
                    self._cache[cache_key]
                ):
                    cached_result = self._cache[cache_key]['result']
                    self.logger.debug(f"Returning cached result: {cached_result}")
                    return cached_result

                self._calculation_count += 1

                # Get current slot availability
                try:
                    available_slots = self.scheduler.get_available_slots()
                except Exception as e:
                    self.logger.error(f"Failed to get available slots: {e}")
                    raise CapacityCalculationError(f"Cannot get slot information: {e}")

                # Update capacity tracking
                self._update_capacity_tracking(available_slots)

                if not available_slots:
                    self.logger.warning("No available slots found")
                    return self.min_concurrent_dags

                # Simple bottleneck calculation: minimum available slots
                bottleneck_capacity = min(available_slots.values())
                bottleneck_resource = self._find_bottleneck_resource(available_slots)

                # Apply utilization factor if we have pending work
                utilization_factor = 1.0
                if flat_jobs is not None:
                    utilization_factor = self._calculate_slot_utilization_factor(
                        available_slots, flat_jobs
                    )
                    self.logger.debug(
                        f"Applying slot utilization factor: {utilization_factor:.3f}"
                    )

                final_capacity = int(bottleneck_capacity * utilization_factor)

                # Apply global bounds
                result = max(
                    self.min_concurrent_dags,
                    min(self.max_concurrent_dags, final_capacity),
                )

                # Store metrics with integer timestamp
                calculation_time = (time.time() - start_time) * 1000  # ms
                self.last_calculation_metrics = CapacityMetrics(
                    calculation_time_ms=calculation_time,
                    bottleneck_capacity=bottleneck_capacity,
                    final_capacity=result,
                    bottleneck_resource=bottleneck_resource,
                    utilization_factor=(
                        utilization_factor if flat_jobs is not None else None
                    ),
                    timestamp=self._get_current_timestamp_ms(),
                )

                # Cache result with integer timestamp
                self._cache[cache_key] = {
                    'result': result,
                    'timestamp': self._get_current_timestamp_ms(),
                }
                self._cleanup_cache()

                utilization_info = (
                    f" (utilization factor: {utilization_factor:.3f})"
                    if flat_jobs is not None
                    else ""
                )
                self.logger.info(
                    f"Calculated max_concurrent_dags: {result} (bottleneck: {bottleneck_capacity} from {bottleneck_resource}){utilization_info}"
                )
                return result

            except Exception as e:
                self.logger.error(f"Error calculating max concurrent DAGs: {e}")
                self._error_count += 1
                fallback = self.min_concurrent_dags
                self.logger.warning(f"Returning fallback value: {fallback}")
                return fallback

    def _calculate_slot_utilization_factor(
        self, available_slots: Dict[str, int], flat_jobs: List[Tuple[str, Any]]
    ) -> float:
        """
        Calculate adjustment factor based on slot utilization for parallel DAG execution.
        """
        if not flat_jobs:
            self.logger.debug(
                "No pending jobs - system is idle, maintaining current concurrency level"
            )
            return 1.0

        # Count pending jobs by slot type
        pending_jobs_by_slot = {}
        for slot_name, work_info in flat_jobs:
            pending_jobs_by_slot[slot_name] = pending_jobs_by_slot.get(slot_name, 0) + 1

        # Analyze each slot type for bottlenecks
        bottleneck_count = 0
        severe_bottleneck_count = 0
        total_slot_types = len(available_slots)

        for slot_type in available_slots:
            available_count = available_slots[slot_type]
            pending_count = pending_jobs_by_slot.get(slot_type, 0)

            if available_count == 0 and pending_count > 0:
                # BOTTLENECK: No slots available but jobs are waiting
                expected_capacity = self._get_expected_capacity(slot_type)

                # Calculate severity based on pending jobs vs expected capacity
                severity_ratio = pending_count / expected_capacity

                if severity_ratio > 1.5:  # More than 1.5x expected capacity waiting
                    severe_bottleneck_count += 1
                    self.logger.debug(
                        f"Severe bottleneck: {slot_type} has {pending_count} pending jobs, "
                        f"0 available, expected capacity: {expected_capacity} "
                        f"(severity ratio: {severity_ratio:.2f})"
                    )
                else:
                    self.logger.debug(
                        f"Bottleneck: {slot_type} has {pending_count} pending jobs, "
                        f"0 available, expected capacity: {expected_capacity} "
                        f"(severity ratio: {severity_ratio:.2f})"
                    )

                bottleneck_count += 1

        # Calculate utilization factor based on bottleneck severity
        if severe_bottleneck_count > 0:
            # Severe bottlenecks - aggressively reduce concurrency
            penalty = min(0.8, severe_bottleneck_count / total_slot_types)
            utilization_factor = max(0.1, 1.0 - penalty)
            self.logger.debug(f"Applied severe bottleneck penalty: {penalty:.3f}")
        elif bottleneck_count > 0:
            # Regular bottlenecks - moderately reduce concurrency
            penalty = min(0.6, bottleneck_count / total_slot_types * 0.5)
            utilization_factor = max(0.3, 1.0 - penalty)
            self.logger.debug(f"Applied bottleneck penalty: {penalty:.3f}")
        else:
            # No bottlenecks - check for high demand
            max_demand_ratio = 0
            for slot_type in available_slots:
                available_count = available_slots[slot_type]
                pending_count = pending_jobs_by_slot.get(slot_type, 0)

                if available_count > 0:
                    demand_ratio = pending_count / available_count
                    max_demand_ratio = max(max_demand_ratio, demand_ratio)

            if max_demand_ratio > 2.0:
                # Very high demand - slightly reduce concurrency
                utilization_factor = 0.8
                self.logger.debug(
                    f"High demand detected (ratio: {max_demand_ratio:.2f})"
                )
            elif max_demand_ratio > 1.0:
                # Moderate demand - slightly reduce concurrency
                utilization_factor = 0.9
                self.logger.debug(
                    f"Moderate demand detected (ratio: {max_demand_ratio:.2f})"
                )
            else:
                # Low demand - system can handle current load
                utilization_factor = 1.0

        self.logger.debug(f"Slot utilization analysis:")
        self.logger.debug(f"  Available slots: {available_slots}")
        self.logger.debug(f"  Max observed capacity: {self._max_observed_capacity}")
        self.logger.debug(f"  Pending jobs by slot: {pending_jobs_by_slot}")
        self.logger.debug(
            f"  Bottlenecks: {bottleneck_count}, Severe: {severe_bottleneck_count}"
        )
        self.logger.debug(f"  Final adjustment factor: {utilization_factor:.3f}")

        return utilization_factor

    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if current_time - entry['timestamp'] > self._cache_ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]

    def get_capacity_tracking_info(self) -> Dict[str, Dict[str, int]]:
        """Get current capacity tracking information for debugging/monitoring."""
        with self._lock:
            return {"max_observed_capacity": self._max_observed_capacity.copy()}

    def reset_capacity_tracking(self):
        """Reset capacity tracking (useful for testing or system resets)."""
        with self._lock:
            self._max_observed_capacity.clear()
            self.logger.info("Reset capacity tracking")

    def get_capacity_analysis(self, flat_jobs: List[Tuple[str, Any]] = None) -> Dict:
        """Get detailed capacity analysis for monitoring/debugging"""
        with self._lock:
            try:
                available_slots = self.scheduler.get_available_slots()

                # Calculate bottleneck capacity
                bottleneck_capacity = (
                    min(available_slots.values()) if available_slots else 0
                )
                bottleneck_resource = self._find_bottleneck_resource(available_slots)

                # Calculate utilization factor if jobs provided
                utilization_factor = None
                if flat_jobs is not None:
                    utilization_factor = self._calculate_slot_utilization_factor(
                        available_slots, flat_jobs
                    )

                # Calculate final capacity
                if utilization_factor is not None:
                    final_capacity = int(bottleneck_capacity * utilization_factor)
                else:
                    final_capacity = bottleneck_capacity

                # Apply bounds
                bounded_capacity = max(
                    self.min_concurrent_dags,
                    min(self.max_concurrent_dags, final_capacity),
                )

                return {
                    "available_slots": available_slots,
                    "max_observed_capacity": self._max_observed_capacity.copy(),
                    "bottleneck_capacity": bottleneck_capacity,
                    "utilization_factor": utilization_factor,
                    "final_capacity": final_capacity,
                    "bounded_capacity": bounded_capacity,
                    "bottleneck_resource": bottleneck_resource,
                    "limits": {
                        "min": self.min_concurrent_dags,
                        "max": self.max_concurrent_dags,
                    },
                    "calculation_count": self._calculation_count,
                    "error_count": self._error_count,
                }
            except Exception as e:
                self.logger.error(f"Error generating capacity analysis: {e}")
                return {"error": str(e)}

    def get_health_status(self) -> Dict:
        """Get health status for monitoring"""
        with self._lock:
            try:
                available_slots = self.scheduler.get_available_slots()
                total_available = sum(available_slots.values())

                # Check for issues
                issues = []
                if total_available == 0:
                    issues.append("No available slots")
                if self._error_count > 0:
                    issues.append(f"Calculation errors: {self._error_count}")

                # Determine status
                if not issues:
                    status = "healthy"
                elif len(issues) == 1 and "errors" in issues[0]:
                    status = "degraded"
                else:
                    status = "unhealthy"

                return {
                    "status": status,
                    "total_available_slots": total_available,
                    "slot_types_count": len(available_slots),
                    "calculation_count": self._calculation_count,
                    "error_count": self._error_count,
                    "cache_size": len(self._cache),
                    "max_observed_slots": len(self._max_observed_capacity),
                    "issues": issues,
                    "timestamp": time.time(),
                }
            except Exception as e:
                return {"status": "error", "error": str(e), "timestamp": time.time()}

    def update_dag_limits(self, min_dags: int = None, max_dags: int = None):
        """Update DAG concurrency limits"""
        with self._lock:
            if min_dags is not None:
                if min_dags < 0:
                    raise BadConfigSource(
                        f"min_concurrent_dags must be >= 0: {min_dags}"
                    )
                self.min_concurrent_dags = min_dags

            if max_dags is not None:
                if max_dags < self.min_concurrent_dags:
                    raise BadConfigSource(
                        f"max_concurrent_dags ({max_dags}) < min_concurrent_dags ({self.min_concurrent_dags})"
                    )
                self.max_concurrent_dags = max_dags

            self._cache.clear()
            self.logger.info(
                f"Updated DAG limits: min={self.min_concurrent_dags}, max={self.max_concurrent_dags}"
            )

    def get_configuration_summary(self) -> Dict:
        """Get complete configuration summary with enhanced CapacityMetrics integration"""
        with self._lock:
            summary = {
                "configuration": {
                    "dag_limits": {
                        "min": self.min_concurrent_dags,
                        "max": self.max_concurrent_dags,
                        "range": self.max_concurrent_dags - self.min_concurrent_dags,
                    },
                    "cache_ttl_seconds": self._cache_ttl_seconds,
                    "scheduler_class": self.scheduler.__class__.__name__,
                },
                "capacity_tracking": {
                    "max_observed_capacity": self._max_observed_capacity.copy(),
                    "tracked_slot_types": len(self._max_observed_capacity),
                    "total_max_observed": sum(self._max_observed_capacity.values()),
                },
                "runtime_metrics": {
                    "calculation_count": self._calculation_count,
                    "error_count": self._error_count,
                    "cache_size": len(self._cache),
                    "error_rate_percent": (
                        self._error_count / max(1, self._calculation_count)
                    )
                    * 100,
                },
            }

            # Rich last calculation info using CapacityMetrics
            if self.last_calculation_metrics:
                summary["last_calculation"] = (
                    self.last_calculation_metrics.to_export_dict("")
                )
                summary["last_calculation"]["age_seconds"] = (
                    time.time() - self.last_calculation_metrics.timestamp
                )
            else:
                summary["last_calculation"] = {"status": "no_calculations_performed"}

            # Current system state
            try:
                current_slots = self.scheduler.get_available_slots()
                summary["current_state"] = {
                    "available_slots": current_slots,
                    "bottleneck_slot": (
                        min(current_slots, key=current_slots.get)
                        if current_slots
                        else None
                    ),
                    "bottleneck_capacity": (
                        min(current_slots.values()) if current_slots else 0
                    ),
                    "total_capacity": sum(current_slots.values()),
                    "theoretical_max_dags": min(
                        self.max_concurrent_dags,
                        min(current_slots.values()) if current_slots else 0,
                    ),
                }
            except Exception as e:
                summary["current_state"] = {"error": str(e)}

            return summary

    def get_metrics_for_export(self) -> Dict:
        """Export metrics in format suitable for monitoring systems"""
        with self._lock:
            metrics = {
                'dag_concurrency_calculations_total': self._calculation_count,
                'dag_concurrency_errors_total': self._error_count,
                'dag_concurrency_cache_size': len(self._cache),
                'dag_concurrency_cache_ttl_seconds': self._cache_ttl_seconds,
                'dag_concurrency_tracked_slot_types': len(self._max_observed_capacity),
            }

            # Add capacity metrics if available
            if self.last_calculation_metrics:
                metrics.update(self.last_calculation_metrics.to_export_dict())
            else:
                # Default values when no calculations performed yet
                metrics.update(
                    {
                        'dag_concurrency_current_capacity': 0,
                        'dag_concurrency_bottleneck_capacity': 0,
                        'dag_concurrency_bottleneck_resource': 'unknown',
                        'dag_concurrency_last_calculation_timestamp': 0,
                        'dag_concurrency_calculation_age_seconds': -1,
                        'dag_concurrency_has_pending_jobs': 0,
                    }
                )

            return metrics

    def reset_configuration(self):
        """Reset configuration to defaults (useful for testing)"""
        with self._lock:
            # Reset to initial values
            self.min_concurrent_dags = 1
            self.max_concurrent_dags = 50

            # Clear state
            self._cache.clear()
            self._max_observed_capacity.clear()
            self._calculation_count = 0
            self._error_count = 0

            self.logger.info("Configuration reset to defaults")
