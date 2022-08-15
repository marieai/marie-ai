from typing import Any


class HealthCheckHandler:
    def check(self, config: Any, **kwargs: Any) -> bool:
        """
        Run a health check
        Args:
            config : health check config

        Returns:
            bool: status of a health check
        """
        raise NotImplementedError()
