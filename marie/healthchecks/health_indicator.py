from enum import Enum


class HealthIndicator(Enum):
    """
    # *
    # * Indicates state of the service being monitored
    # * <ul>
    # * <li> critical - Critical but still responding</li>
    # * <li> failure  - Service is unresponsive</li>
    # * <li> healthy  - Service is operational</li>
    # * </ul>
    #
    """

    HEALTHY = 0
    FAILURE = 1
    CRITICAL = 2
