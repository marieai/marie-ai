import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional


class CapacityCalculationError(Exception):
    """Custom exception for capacity calculation errors"""

    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""

    pass


@dataclass
class CapacityMetrics:
    """Metrics for monitoring capacity calculations"""

    calculation_time_ms: float
    weighted_capacity: int
    conservative_capacity: int
    final_capacity: int
    bottleneck_resource: Optional[str]
    timestamp: float


class DagConcurrencyManager:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.RLock()  # Thread safety

        # Configuration validation
        self._validate_scheduler()

        # Per-slot multipliers (only define known ones)
        self.slot_multipliers = {
            'annotator_llm': 2,
            'annotator_embeddings': 3,
            'annotator_parser': 2,
            'annotator_table': 1,
            'annotator_table_parser': 2,
        }

        # Resource weights (only define known ones, should sum to 1.0)
        self.resource_weights = {
            'annotator_llm': 0.4,
            'annotator_embeddings': 0.3,
            'annotator_parser': 0.2,
            'annotator_table': 0.08,
            'annotator_table_parser': 0.02,
        }

        # Global DAG limits with validation
        self.min_concurrent_dags = 1
        self.max_concurrent_dags = 50

        # Strategy weights for combining approaches
        self.weighted_strategy_weight = 0.6
        self.conservative_strategy_weight = 0.4

        # Default values for unknown slot types
        self.default_multiplier = 2
        self.default_weight = 0.0

        # Monitoring and metrics
        self.last_calculation_metrics: Optional[CapacityMetrics] = None
        self._calculation_count = 0
        self._error_count = 0

        # Cache for performance (with TTL)
        self._cache = {}
        self._cache_ttl_seconds = 10  # Cache results for 10 seconds

        # Initialize and validate
        try:
            self._ensure_key_consistency()
            self._validate_configuration()
            self.logger.info("DagConcurrencyManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize DagConcurrencyManager: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")

    def _validate_scheduler(self):
        """Validate that scheduler has required methods"""
        if not hasattr(self.scheduler, 'get_available_slots'):
            raise ConfigurationError("Scheduler must have 'get_available_slots' method")

    def _validate_configuration(self):
        """Validate configuration values"""
        # Validate multipliers
        for slot_type, multiplier in self.slot_multipliers.items():
            if not isinstance(multiplier, (int, float)) or multiplier <= 0:
                raise ConfigurationError(
                    f"Invalid multiplier for {slot_type}: {multiplier}"
                )

        # Validate weights
        for slot_type, weight in self.resource_weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ConfigurationError(f"Invalid weight for {slot_type}: {weight}")

        # Validate limits
        if self.min_concurrent_dags < 0:
            raise ConfigurationError(
                f"min_concurrent_dags must be >= 0: {self.min_concurrent_dags}"
            )
        if self.max_concurrent_dags < self.min_concurrent_dags:
            raise ConfigurationError(
                f"max_concurrent_dags ({self.max_concurrent_dags}) < min_concurrent_dags ({self.min_concurrent_dags})"
            )

        # Validate strategy weights
        total_strategy_weight = (
            self.weighted_strategy_weight + self.conservative_strategy_weight
        )
        if (
            abs(total_strategy_weight - 1.0) > 0.001
        ):  # Allow small floating point errors
            raise ConfigurationError(
                f"Strategy weights must sum to 1.0, got {total_strategy_weight}"
            )

    def _get_cache_key(self) -> str:
        """Generate cache key based on current state"""
        try:
            available_slots = self.scheduler.get_available_slots()
            return f"{hash(tuple(sorted(available_slots.items())))}"
        except Exception:
            return f"error_{time.time()}"  # Unique key if slots unavailable

    def _is_cache_valid(self, cache_entry) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry['timestamp'] < self._cache_ttl_seconds

    def _ensure_key_consistency(self):
        """Ensure slot_multipliers and resource_weights have consistent keys"""
        with self._lock:
            try:
                # Get all slot types that are explicitly configured
                configured_slot_types = set()
                configured_slot_types.update(self.slot_multipliers.keys())
                configured_slot_types.update(self.resource_weights.keys())

                # Ensure all explicitly configured slot types have both multipliers and weights
                for slot_type in configured_slot_types:
                    if slot_type not in self.slot_multipliers:
                        self.slot_multipliers[slot_type] = self.default_multiplier
                        self.logger.warning(f"Added default multiplier for {slot_type}")
                    if slot_type not in self.resource_weights:
                        self.resource_weights[slot_type] = self.default_weight
                        self.logger.warning(f"Added default weight for {slot_type}")

                # Normalize weights to sum to 1.0
                self._normalize_resource_weights()

            except Exception as e:
                self.logger.error(f"Error ensuring key consistency: {e}")
                raise CapacityCalculationError(f"Key consistency check failed: {e}")

    def _normalize_resource_weights(self):
        """Normalize resource weights to sum to 1.0"""
        try:
            total_weight = sum(self.resource_weights.values())
            if total_weight > 0:
                for slot_type in self.resource_weights:
                    self.resource_weights[slot_type] /= total_weight
            else:
                # If all weights are 0, distribute equally among configured types
                if self.resource_weights:
                    equal_weight = 1.0 / len(self.resource_weights)
                    for slot_type in self.resource_weights:
                        self.resource_weights[slot_type] = equal_weight
                    self.logger.warning("All weights were 0, distributed equally")
        except Exception as e:
            self.logger.error(f"Error normalizing weights: {e}")
            raise CapacityCalculationError(f"Weight normalization failed: {e}")

    def calculate_weighted_capacity(self) -> int:
        """Option 2: Weighted resource allocation"""
        try:
            available_slots = self.scheduler.get_available_slots()

            if not available_slots:
                self.logger.warning("No available slots found")
                return 0

            weighted_capacity = 0
            for slot_type, slots in available_slots.items():
                if slots < 0:
                    self.logger.warning(f"Negative slot count for {slot_type}: {slots}")
                    continue

                multiplier = self.slot_multipliers.get(
                    slot_type, self.default_multiplier
                )
                weight = self.resource_weights.get(slot_type, self.default_weight)
                contribution = slots * multiplier * weight
                weighted_capacity += contribution

                self.logger.debug(
                    f"Weighted calc: {slot_type} = {slots} × {multiplier} × {weight:.3f} = {contribution:.2f}"
                )

            result = int(weighted_capacity)
            self.logger.debug(f"Total weighted capacity: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating weighted capacity: {e}")
            self._error_count += 1
            raise CapacityCalculationError(f"Weighted capacity calculation failed: {e}")

    def calculate_conservative_capacity(self) -> int:
        """Option 3: Conservative bottleneck approach"""
        try:
            available_slots = self.scheduler.get_available_slots()

            if not available_slots:
                self.logger.warning("No available slots found")
                return 0

            min_capacity = float('inf')
            bottleneck_resource = None

            for slot_type, slots in available_slots.items():
                if slots > 0:  # Only consider available resources
                    multiplier = self.slot_multipliers.get(
                        slot_type, self.default_multiplier
                    )
                    capacity = slots * multiplier

                    if capacity < min_capacity:
                        min_capacity = capacity
                        bottleneck_resource = slot_type

                    self.logger.debug(
                        f"Conservative calc: {slot_type} = {slots} × {multiplier} = {capacity}"
                    )

            result = int(min_capacity) if min_capacity != float('inf') else 0

            if bottleneck_resource:
                self.logger.debug(
                    f"Bottleneck resource: {bottleneck_resource}, capacity: {result}"
                )

            return result

        except Exception as e:
            self.logger.error(f"Error calculating conservative capacity: {e}")
            self._error_count += 1
            raise CapacityCalculationError(
                f"Conservative capacity calculation failed: {e}"
            )

    def calculate_max_concurrent_dags(self) -> int:
        """Combined approach using weighted average + conservative bottleneck with caching"""
        with self._lock:
            start_time = time.time()

            try:
                # Check cache first
                cache_key = self._get_cache_key()
                if cache_key in self._cache and self._is_cache_valid(
                    self._cache[cache_key]
                ):
                    cached_result = self._cache[cache_key]['result']
                    self.logger.debug(f"Returning cached result: {cached_result}")
                    return cached_result

                self._calculation_count += 1

                # Calculate using both approaches
                weighted_capacity = self.calculate_weighted_capacity()
                conservative_capacity = self.calculate_conservative_capacity()

                # Combine using weighted average
                combined_capacity = (
                    weighted_capacity * self.weighted_strategy_weight
                    + conservative_capacity * self.conservative_strategy_weight
                )

                final_capacity = int(combined_capacity)

                # Apply global bounds
                result = max(
                    self.min_concurrent_dags,
                    min(self.max_concurrent_dags, final_capacity),
                )

                # Store metrics
                calculation_time = (time.time() - start_time) * 1000  # ms
                self.last_calculation_metrics = CapacityMetrics(
                    calculation_time_ms=calculation_time,
                    weighted_capacity=weighted_capacity,
                    conservative_capacity=conservative_capacity,
                    final_capacity=result,
                    bottleneck_resource=self._find_bottleneck_resource(),
                    timestamp=time.time(),
                )

                # Cache result
                self._cache[cache_key] = {'result': result, 'timestamp': time.time()}

                # Clean old cache entries
                self._cleanup_cache()

                self.logger.info(
                    f"Calculated max_concurrent_dags: {result} (weighted: {weighted_capacity}, conservative: {conservative_capacity})"
                )
                return result

            except Exception as e:
                self.logger.error(f"Error calculating max concurrent DAGs: {e}")
                self._error_count += 1
                # Return safe fallback
                fallback = self.min_concurrent_dags
                self.logger.warning(f"Returning fallback value: {fallback}")
                return fallback

    def _find_bottleneck_resource(self) -> Optional[str]:
        """Find the current bottleneck resource"""
        try:
            available_slots = self.scheduler.get_available_slots()
            min_capacity = float('inf')
            bottleneck = None

            for slot_type, slots in available_slots.items():
                if slots > 0:
                    multiplier = self.slot_multipliers.get(
                        slot_type, self.default_multiplier
                    )
                    capacity = slots * multiplier
                    if capacity < min_capacity:
                        min_capacity = capacity
                        bottleneck = slot_type

            return bottleneck
        except Exception:
            return None

    def _cleanup_cache(self):
        """Remove expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if current_time - entry['timestamp'] > self._cache_ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]
        except Exception as e:
            self.logger.warning(f"Error cleaning cache: {e}")

    def get_capacity_analysis(self) -> dict:
        """Get detailed breakdown of capacity calculations"""
        with self._lock:
            try:
                available_slots = self.scheduler.get_available_slots()

                weighted_capacity = self.calculate_weighted_capacity()
                conservative_capacity = self.calculate_conservative_capacity()

                combined_raw = (
                    weighted_capacity * self.weighted_strategy_weight
                    + conservative_capacity * self.conservative_strategy_weight
                )

                final_result = max(
                    self.min_concurrent_dags,
                    min(self.max_concurrent_dags, int(combined_raw)),
                )

                # Detailed breakdown per slot type
                slot_breakdown = {}
                for slot_type, slots in available_slots.items():
                    multiplier = self.slot_multipliers.get(
                        slot_type, self.default_multiplier
                    )
                    weight = self.resource_weights.get(slot_type, self.default_weight)
                    is_using_defaults = {
                        'multiplier': slot_type not in self.slot_multipliers,
                        'weight': slot_type not in self.resource_weights,
                    }

                    slot_breakdown[slot_type] = {
                        'slots': slots,
                        'multiplier': multiplier,
                        'weight': weight,
                        'capacity': slots * multiplier,
                        'weighted_contribution': slots * multiplier * weight,
                        'using_defaults': is_using_defaults,
                    }

                return {
                    'available_slots': available_slots,
                    'slot_breakdown': slot_breakdown,
                    'calculations': {
                        'weighted_average': {
                            'value': weighted_capacity,
                            'weight': self.weighted_strategy_weight,
                            'contribution': weighted_capacity
                            * self.weighted_strategy_weight,
                        },
                        'conservative': {
                            'value': conservative_capacity,
                            'weight': self.conservative_strategy_weight,
                            'contribution': conservative_capacity
                            * self.conservative_strategy_weight,
                        },
                    },
                    'combined_raw': combined_raw,
                    'final_result': final_result,
                    'bounds_applied': {
                        'min': self.min_concurrent_dags,
                        'max': self.max_concurrent_dags,
                        'was_bounded': final_result != int(combined_raw),
                    },
                    'defaults': {
                        'default_multiplier': self.default_multiplier,
                        'default_weight': self.default_weight,
                    },
                    'metrics': {
                        'calculation_count': self._calculation_count,
                        'error_count': self._error_count,
                        'cache_size': len(self._cache),
                        'last_calculation': (
                            self.last_calculation_metrics.__dict__
                            if self.last_calculation_metrics
                            else None
                        ),
                    },
                }

            except Exception as e:
                self.logger.error(f"Error generating capacity analysis: {e}")
                raise CapacityCalculationError(f"Analysis generation failed: {e}")

    def get_health_status(self) -> dict:
        """Get health status of the concurrency manager"""
        try:
            error_rate = self._error_count / max(self._calculation_count, 1)

            status = {
                'healthy': error_rate < 0.1,  # Less than 10% error rate
                'error_rate': error_rate,
                'total_calculations': self._calculation_count,
                'total_errors': self._error_count,
                'cache_size': len(self._cache),
                'last_calculation_time': (
                    self.last_calculation_metrics.timestamp
                    if self.last_calculation_metrics
                    else None
                ),
            }

            if self.last_calculation_metrics:
                status['last_calculation_duration_ms'] = (
                    self.last_calculation_metrics.calculation_time_ms
                )
                status['bottleneck_resource'] = (
                    self.last_calculation_metrics.bottleneck_resource
                )

            return status

        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return {'healthy': False, 'error': str(e)}

    def update_strategy_balance(
        self, weighted_weight: float, conservative_weight: float
    ):
        """Update the balance between weighted and conservative strategies"""
        with self._lock:
            try:
                # Validate inputs
                if weighted_weight < 0 or conservative_weight < 0:
                    raise ConfigurationError("Strategy weights must be non-negative")

                # Normalize to ensure they sum to 1.0
                total = weighted_weight + conservative_weight
                if total == 0:
                    raise ConfigurationError("Strategy weights cannot both be zero")

                self.weighted_strategy_weight = weighted_weight / total
                self.conservative_strategy_weight = conservative_weight / total

                self.logger.info(
                    f"Updated strategy balance: weighted={self.weighted_strategy_weight:.3f}, conservative={self.conservative_strategy_weight:.3f}"
                )

                # Clear cache as strategy changed
                self._cache.clear()

            except Exception as e:
                self.logger.error(f"Error updating strategy balance: {e}")
                raise ConfigurationError(f"Failed to update strategy balance: {e}")

    def update_resource_weights(self, **weights):
        """Update resource weights for weighted calculation"""
        with self._lock:
            try:
                # Validate all weights are non-negative
                for slot_type, weight in weights.items():
                    if not isinstance(weight, (int, float)) or weight < 0:
                        raise ConfigurationError(
                            f"Invalid weight for {slot_type}: {weight}"
                        )

                # Update weights
                for slot_type, weight in weights.items():
                    self.resource_weights[slot_type] = weight
                    self.logger.debug(f"Updated weight for {slot_type}: {weight}")

                # Normalize weights and ensure consistency
                self._normalize_resource_weights()
                self._ensure_key_consistency()

                self.logger.info(
                    f"Updated resource weights for {len(weights)} slot types"
                )

                # Clear cache as weights changed
                self._cache.clear()

            except Exception as e:
                self.logger.error(f"Error updating resource weights: {e}")
                raise ConfigurationError(f"Failed to update resource weights: {e}")

    def update_slot_multiplier(self, slot_type: str, multiplier: int):
        """Update multiplier for specific slot type"""
        with self._lock:
            try:
                # Validate inputs
                if not isinstance(slot_type, str) or not slot_type.strip():
                    raise ConfigurationError("Slot type must be a non-empty string")

                if not isinstance(multiplier, (int, float)) or multiplier <= 0:
                    raise ConfigurationError(
                        f"Multiplier must be positive, got: {multiplier}"
                    )

                # Update multiplier
                old_multiplier = self.slot_multipliers.get(
                    slot_type, self.default_multiplier
                )
                self.slot_multipliers[slot_type] = int(multiplier)

                self.logger.info(
                    f"Updated multiplier for {slot_type}: {old_multiplier} -> {multiplier}"
                )

                # Ensure consistency after update
                self._ensure_key_consistency()

                # Clear cache as multiplier changed
                self._cache.clear()

            except Exception as e:
                self.logger.error(f"Error updating slot multiplier: {e}")
                raise ConfigurationError(f"Failed to update slot multiplier: {e}")

    def add_slot_type_config(
        self, slot_type: str, multiplier: int = None, weight: float = None
    ):
        """Add configuration for a new slot type"""
        with self._lock:
            try:
                # Validate inputs
                if not isinstance(slot_type, str) or not slot_type.strip():
                    raise ConfigurationError("Slot type must be a non-empty string")

                if multiplier is not None:
                    if not isinstance(multiplier, (int, float)) or multiplier <= 0:
                        raise ConfigurationError(
                            f"Multiplier must be positive, got: {multiplier}"
                        )
                    self.slot_multipliers[slot_type] = int(multiplier)
                    self.logger.debug(f"Set multiplier for {slot_type}: {multiplier}")

                if weight is not None:
                    if not isinstance(weight, (int, float)) or weight < 0:
                        raise ConfigurationError(
                            f"Weight must be non-negative, got: {weight}"
                        )
                    self.resource_weights[slot_type] = float(weight)
                    self.logger.debug(f"Set weight for {slot_type}: {weight}")

                # Normalize weights and ensure consistency
                if weight is not None:
                    self._normalize_resource_weights()
                self._ensure_key_consistency()

                self.logger.info(f"Added slot type configuration: {slot_type}")

                # Clear cache as configuration changed
                self._cache.clear()

            except Exception as e:
                self.logger.error(f"Error adding slot type config: {e}")
                raise ConfigurationError(f"Failed to add slot type config: {e}")

    def get_slot_types_status(self) -> dict:
        """Get status of all slot types and their configuration"""
        with self._lock:
            try:
                available_slots = self.scheduler.get_available_slots()

                status = {}
                # Only show slot types that actually exist in the scheduler
                for slot_type, slots in available_slots.items():
                    is_multiplier_configured = slot_type in self.slot_multipliers
                    is_weight_configured = slot_type in self.resource_weights

                    status[slot_type] = {
                        'available_slots': slots,
                        'multiplier': self.slot_multipliers.get(
                            slot_type, self.default_multiplier
                        ),
                        'weight': self.resource_weights.get(
                            slot_type, self.default_weight
                        ),
                        'using_default_multiplier': not is_multiplier_configured,
                        'using_default_weight': not is_weight_configured,
                        'fully_configured': is_multiplier_configured
                        and is_weight_configured,
                    }

                self.logger.debug(f"Retrieved status for {len(status)} slot types")
                return status

            except Exception as e:
                self.logger.error(f"Error getting slot types status: {e}")
                raise CapacityCalculationError(f"Failed to get slot types status: {e}")

    def update_defaults(
        self, default_multiplier: int = None, default_weight: float = None
    ):
        """Update default values for unconfigured slot types"""
        with self._lock:
            try:
                if default_multiplier is not None:
                    if (
                        not isinstance(default_multiplier, (int, float))
                        or default_multiplier <= 0
                    ):
                        raise ConfigurationError(
                            f"Default multiplier must be positive, got: {default_multiplier}"
                        )
                    old_default = self.default_multiplier
                    self.default_multiplier = int(default_multiplier)
                    self.logger.info(
                        f"Updated default multiplier: {old_default} -> {default_multiplier}"
                    )

                if default_weight is not None:
                    if (
                        not isinstance(default_weight, (int, float))
                        or default_weight < 0
                    ):
                        raise ConfigurationError(
                            f"Default weight must be non-negative, got: {default_weight}"
                        )
                    old_default = self.default_weight
                    self.default_weight = float(default_weight)
                    self.logger.info(
                        f"Updated default weight: {old_default} -> {default_weight}"
                    )

                # Clear cache as defaults changed
                self._cache.clear()

            except Exception as e:
                self.logger.error(f"Error updating defaults: {e}")
                raise ConfigurationError(f"Failed to update defaults: {e}")

    def update_dag_limits(
        self, min_concurrent_dags: int = None, max_concurrent_dags: int = None
    ):
        """Update global DAG concurrency limits"""
        with self._lock:
            try:
                if min_concurrent_dags is not None:
                    if (
                        not isinstance(min_concurrent_dags, int)
                        or min_concurrent_dags < 0
                    ):
                        raise ConfigurationError(
                            f"min_concurrent_dags must be non-negative integer, got: {min_concurrent_dags}"
                        )
                    self.min_concurrent_dags = min_concurrent_dags
                    self.logger.info(
                        f"Updated min_concurrent_dags: {min_concurrent_dags}"
                    )

                if max_concurrent_dags is not None:
                    if (
                        not isinstance(max_concurrent_dags, int)
                        or max_concurrent_dags < 1
                    ):
                        raise ConfigurationError(
                            f"max_concurrent_dags must be positive integer, got: {max_concurrent_dags}"
                        )
                    self.max_concurrent_dags = max_concurrent_dags
                    self.logger.info(
                        f"Updated max_concurrent_dags: {max_concurrent_dags}"
                    )

                # Validate that max >= min
                if self.max_concurrent_dags < self.min_concurrent_dags:
                    raise ConfigurationError(
                        f"max_concurrent_dags ({self.max_concurrent_dags}) < min_concurrent_dags ({self.min_concurrent_dags})"
                    )

                # Clear cache as limits changed
                self._cache.clear()

            except Exception as e:
                self.logger.error(f"Error updating DAG limits: {e}")
                raise ConfigurationError(f"Failed to update DAG limits: {e}")

    def get_configuration_summary(self) -> dict:
        """Get a summary of current configuration"""
        with self._lock:
            try:
                return {
                    'strategy_weights': {
                        'weighted': self.weighted_strategy_weight,
                        'conservative': self.conservative_strategy_weight,
                    },
                    'dag_limits': {
                        'min_concurrent_dags': self.min_concurrent_dags,
                        'max_concurrent_dags': self.max_concurrent_dags,
                    },
                    'defaults': {
                        'default_multiplier': self.default_multiplier,
                        'default_weight': self.default_weight,
                    },
                    'configured_slot_types': {
                        'multipliers': dict(self.slot_multipliers),
                        'weights': dict(self.resource_weights),
                    },
                    'cache_settings': {
                        'ttl_seconds': self._cache_ttl_seconds,
                        'current_size': len(self._cache),
                    },
                }
            except Exception as e:
                self.logger.error(f"Error getting configuration summary: {e}")
                return {'error': str(e)}

    def reset_configuration(self):
        """Reset configuration to defaults"""
        with self._lock:
            try:
                # Reset to initial configuration
                self.slot_multipliers = {
                    'annotator_llm': 2,
                    'annotator_embeddings': 3,
                    'annotator_parser': 2,
                    'annotator_table': 1,
                    'annotator_table_parser': 2,
                }

                self.resource_weights = {
                    'annotator_llm': 0.4,
                    'annotator_embeddings': 0.3,
                    'annotator_parser': 0.2,
                    'annotator_table': 0.08,
                    'annotator_table_parser': 0.02,
                }

                self.weighted_strategy_weight = 0.6
                self.conservative_strategy_weight = 0.4
                self.min_concurrent_dags = 1
                self.max_concurrent_dags = 50
                self.default_multiplier = 2
                self.default_weight = 0.0

                # Clear cache and ensure consistency
                self._cache.clear()
                self._ensure_key_consistency()

                self.logger.info("Configuration reset to defaults")

            except Exception as e:
                self.logger.error(f"Error resetting configuration: {e}")
                raise ConfigurationError(f"Failed to reset configuration: {e}")
