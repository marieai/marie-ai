"""
Polling sensor implementation.

Polling sensors periodically check an external API for changes,
triggering jobs when specific conditions are detected.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.exceptions import SensorConfigError, SensorEvaluationError
from marie.sensors.registry import register_sensor
from marie.sensors.types import SensorResult, SensorType


@register_sensor(SensorType.POLLING)
class PollingSensor(BaseSensor):
    """
    External API polling sensor.

    This sensor periodically makes HTTP requests to an external API
    and evaluates the response to determine if a job should be triggered.

    Configuration:
        url: str - API URL to poll (required)
        method: str - HTTP method (default: "GET")
        headers: dict - HTTP headers to include
        body: dict - Request body (for POST/PUT)
        response_check: str - JavaScript/Python expression to evaluate response
        change_detection: str - How to detect changes:
            - "any": Fire on any successful response
            - "changed": Fire only when response differs from last poll
            - "condition": Fire when response_check returns True/non-empty

    Cursor:
        Hash of the last response (for change detection)

    Run Key:
        polling:{sensor_id}:{response_hash}:{timestamp}
    """

    sensor_type = SensorType.POLLING

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.url: str = self.get_config_value("url", "")
        self.method: str = self.get_config_value("method", "GET").upper()
        self.headers: Dict[str, str] = self.get_config_value("headers", {})
        self.body: Optional[Dict[str, Any]] = self.get_config_value("body")
        self.response_check: Optional[str] = self.get_config_value("response_check")
        self.change_detection: str = self.get_config_value(
            "change_detection", "changed"
        )
        self.timeout_seconds: int = self.get_config_value("timeout_seconds", 30)

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the polling sensor by making an HTTP request.

        The http_client should be provided in context.resources.
        """
        if not self.url:
            return SensorResult.skip("No URL configured", cursor=context.cursor)

        # Get HTTP client from resources
        http_client = context.resources.get("http_client")
        if http_client is None:
            # Try to create a simple client if aiohttp is available
            try:
                import aiohttp

                http_client = aiohttp.ClientSession()
                context.resources["http_client"] = http_client
                context.resources["_owns_http_client"] = True
            except ImportError:
                return SensorResult.skip(
                    "No HTTP client available (aiohttp not installed)",
                    cursor=context.cursor,
                )

        try:
            response_data = await self._make_request(http_client)
        except Exception as e:
            context.log_error(f"HTTP request failed: {e}")
            raise SensorEvaluationError(
                f"Failed to poll {self.url}: {e}",
                sensor_id=self.sensor_id,
                cause=e,
            )
        finally:
            # Clean up if we created the client
            if context.resources.get("_owns_http_client"):
                await http_client.close()
                del context.resources["http_client"]
                del context.resources["_owns_http_client"]

        # Calculate response hash for change detection
        response_hash = self._hash_response(response_data)

        # Apply change detection logic
        should_fire, skip_reason = self._check_should_fire(
            response_data, response_hash, context.cursor
        )

        if not should_fire:
            return SensorResult.skip(skip_reason, cursor=response_hash)

        context.log_info(f"Polling detected trigger condition at {self.url}")

        # Generate run key
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        run_key = self.build_run_key(
            "polling", self.sensor_id, response_hash[:16], timestamp
        )

        return SensorResult.fire(
            run_key=run_key,
            job_name=self.target_job_name,
            dag_id=self.target_dag_id,
            run_config={
                "url": self.url,
                "response": response_data,
                "response_hash": response_hash,
                "polled_at": datetime.now(timezone.utc).isoformat(),
            },
            tags={
                "trigger": "polling",
                "sensor_id": self.sensor_id,
            },
            cursor=response_hash,
        )

    async def _make_request(self, http_client) -> Any:
        """Make the HTTP request and return response data."""
        import aiohttp

        kwargs = {
            "headers": self.headers,
            "timeout": aiohttp.ClientTimeout(total=self.timeout_seconds),
        }

        if self.body and self.method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = self.body

        async with http_client.request(self.method, self.url, **kwargs) as response:
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return await response.json()
            else:
                return await response.text()

    def _hash_response(self, response_data: Any) -> str:
        """Create a hash of the response for change detection."""
        if isinstance(response_data, dict):
            data_str = json.dumps(response_data, sort_keys=True)
        else:
            data_str = str(response_data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _check_should_fire(
        self, response_data: Any, response_hash: str, last_cursor: Optional[str]
    ) -> tuple[bool, str]:
        """
        Check if the sensor should fire based on change detection mode.

        Returns: (should_fire, skip_reason)
        """
        if self.change_detection == "any":
            # Fire on any successful response
            return True, ""

        elif self.change_detection == "changed":
            # Fire only if response changed
            if last_cursor and response_hash == last_cursor:
                return False, "Response unchanged since last poll"
            return True, ""

        elif self.change_detection == "condition":
            # Fire based on response_check expression
            if not self.response_check:
                return True, ""

            try:
                result = self._evaluate_response_check(response_data)
                if not result:
                    return False, f"Condition not met: {self.response_check}"
                return True, ""
            except Exception as e:
                return False, f"Condition evaluation failed: {e}"

        else:
            # Unknown mode - fire to be safe
            return True, ""

    def _evaluate_response_check(self, response_data: Any) -> bool:
        """
        Evaluate the response_check expression.

        The expression is evaluated with 'response' as the response data.
        Returns True if the result is truthy.
        """
        # Simple expression evaluation
        # In production, consider using a sandboxed evaluator
        local_vars = {"response": response_data}

        # Handle simple field checks
        if "." in self.response_check and "==" not in self.response_check:
            # Simple path like "response.data.items"
            parts = self.response_check.replace("response.", "").split(".")
            value = response_data
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, list) and part.isdigit():
                    value = value[int(part)]
                else:
                    return False
            return bool(value)

        # For more complex expressions, use eval (with caution)
        # In production, use a proper expression parser
        try:
            result = eval(self.response_check, {"__builtins__": {}}, local_vars)
            return bool(result)
        except Exception:
            return False

    def validate_config(self) -> None:
        """Validate polling sensor configuration."""
        if not self.url:
            raise SensorConfigError(
                "Polling sensor requires 'url' configuration", field="url"
            )

        valid_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
        if self.method not in valid_methods:
            raise SensorConfigError(
                f"Invalid HTTP method '{self.method}'. "
                f"Must be one of: {valid_methods}",
                field="method",
            )

        valid_modes = ["any", "changed", "condition"]
        if self.change_detection not in valid_modes:
            raise SensorConfigError(
                f"Invalid change_detection '{self.change_detection}'. "
                f"Must be one of: {valid_modes}",
                field="change_detection",
            )
