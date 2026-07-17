"""Lightweight HTTP client for Marie Gateway (job-centric API)."""

import json
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config


class MarieClientError(Exception):
    """Raised when Marie client operations fail."""

    pass


class MarieClient:
    """
    Lightweight HTTP client for Marie Gateway.

    This client uses the job-centric API where all document processing
    goes through job submission via /api/v1/invoke endpoint.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize Marie client.

        Args:
            base_url: Marie gateway URL (defaults to Config.MARIE_BASE_URL)
            api_key: API key for authentication (defaults to Config.MARIE_API_KEY)
            timeout: Request timeout in seconds (defaults to Config.REQUEST_TIMEOUT)
        """
        self.base_url = (base_url or Config.MARIE_BASE_URL).rstrip("/")
        self.api_key = api_key or Config.MARIE_API_KEY
        self.timeout = timeout or Config.REQUEST_TIMEOUT

        if not self.api_key:
            raise MarieClientError("API key is required")

        self._client = httpx.AsyncClient(timeout=self.timeout)

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
    )
    async def submit_job(
        self, queue_name: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit a job via /api/v1/invoke endpoint.

        This is the PRIMARY method for all document processing.

        Args:
            queue_name: Queue name ('extract', 'gen5_extract', etc.)
            metadata: Job metadata (uri, ref_id, planner, etc.)

        Returns:
            Response with job_id

        Raises:
            MarieClientError: If submission fails
        """
        payload = {
            "invoke_action": {
                "action_type": "command",
                "api_key": self.api_key,
                "command": "job",
                "action": "submit",
                "name": queue_name,
                "metadata": metadata,
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/v1/invoke", json=payload, headers=headers
            )
            response.raise_for_status()

            # Response format: {"header": {}, "parameters": {"job_id": "..."}, "data": None}
            result = response.json()
            parameters = result.get("parameters", {})

            if "job_id" not in parameters:
                raise MarieClientError(f"No job_id in response: {json.dumps(result)}")

            return parameters

        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e
        except Exception as e:
            raise MarieClientError(f"Job submission failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
    )
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status from /api/jobs/{job_id}.

        Args:
            job_id: Job ID to query

        Returns:
            Job status information

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
    )
    async def list_jobs(self, state: Optional[str] = None) -> Dict[str, Any]:
        """
        List jobs from /api/jobs or /api/jobs/{state}.

        Args:
            state: Optional state filter (e.g., 'active', 'completed', 'failed')

        Returns:
            List of jobs

        Raises:
            MarieClientError: If request fails
        """
        try:
            url = f"{self.base_url}/api/jobs"
            if state:
                url = f"{url}/{state}"

            response = await self._client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def stop_job(self, job_id: str) -> Dict[str, Any]:
        """
        Stop job via /api/jobs/{job_id}/stop.

        Args:
            job_id: Job ID to stop

        Returns:
            Response confirmation

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/jobs/{job_id}/stop")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def delete_job(self, job_id: str) -> Dict[str, Any]:
        """
        Delete job via DELETE /api/jobs/{job_id}.

        Args:
            job_id: Job ID to delete

        Returns:
            Response confirmation

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.delete(f"{self.base_url}/api/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def get_deployments(self) -> Dict[str, Any]:
        """
        Get deployments from /api/deployments.

        Returns:
            Deployment information

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/deployments")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def get_capacity(self) -> Dict[str, Any]:
        """
        Get capacity from /api/capacity.

        Returns:
            Capacity and slot information

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/capacity")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def get_debug_info(self) -> Dict[str, Any]:
        """
        Get scheduler debug info from /api/debug.

        Returns:
            Debug information

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/debug")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check via /check endpoint.

        Returns:
            Health status

        Raises:
            MarieClientError: If request fails
        """
        try:
            response = await self._client.get(f"{self.base_url}/check?text=health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise MarieClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise MarieClientError(f"Request failed: {str(e)}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
