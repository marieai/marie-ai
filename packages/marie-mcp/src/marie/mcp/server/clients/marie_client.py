"""Lightweight HTTP client for the Marie Gateway job API."""

from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import Config


class MarieClientError(Exception):
    """Raised when a gateway operation fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _transient_read_error(error: BaseException) -> bool:
    return isinstance(error, httpx.TransportError) or (
        isinstance(error, httpx.HTTPStatusError)
        and error.response.status_code in {429, 500, 502, 503, 504}
    )


class MarieClient:
    """Authenticated asynchronous gateway client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or Config.MARIE_BASE_URL).rstrip('/')
        self.api_key = api_key or Config.MARIE_API_KEY
        self.timeout = Config.REQUEST_TIMEOUT if timeout is None else timeout
        if not self.api_key:
            raise MarieClientError('API key is required')
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={'Authorization': f'Bearer {self.api_key}'},
        )

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as error:
            raise MarieClientError('Gateway returned invalid JSON') from error
        if not isinstance(result, dict):
            raise MarieClientError('Gateway response must be a JSON object')
        parameters = result.get('parameters')
        for body in (result, parameters):
            if isinstance(body, dict) and (
                str(body.get('status', '')).lower() in {'error', 'failed'}
                or body.get('error')
                or body.get('exception')
            ):
                message = (
                    body.get('message')
                    or body.get('msg')
                    or body.get('error')
                    or 'Gateway operation failed'
                )
                raise MarieClientError(str(message))
        return result

    async def submit_job(
        self, queue_name: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit once; a transport failure can leave acceptance unknown."""
        payload = {
            'parameters': {
                'invoke_action': {
                    'action_type': 'command',
                    'api_key': self.api_key,
                    'command': 'job',
                    'action': 'submit',
                    'name': queue_name,
                    'metadata': metadata,
                }
            }
        }
        try:
            response = await self._client.post(
                f'{self.base_url}/api/v1/invoke', json=payload
            )
            result = self._decode(response)
        except httpx.HTTPStatusError as error:
            raise MarieClientError(
                f'HTTP {error.response.status_code}: submission failed; reconcile job state before resubmitting'
            ) from error
        except httpx.RequestError as error:
            raise MarieClientError(
                'Submission outcome unknown after a request failure; reconcile job state before resubmitting'
            ) from error
        parameters = result.get('parameters')
        if (
            not isinstance(parameters, dict)
            or not isinstance(parameters.get('job_id'), str)
            or not parameters['job_id']
        ):
            raise MarieClientError(
                'No job_id in gateway response; reconcile job state before resubmitting'
            )
        return parameters

    @retry(
        retry=retry_if_exception(_transient_read_error),
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
        reraise=True,
    )
    async def _read(self, path: str, params: dict[str, str] | None) -> dict[str, Any]:
        response = await self._client.get(f'{self.base_url}{path}', params=params)
        return self._decode(response)

    async def _get(
        self, path: str, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        try:
            return await self._read(path, params)
        except httpx.HTTPStatusError as error:
            raise MarieClientError(
                f'HTTP {error.response.status_code}: gateway request failed',
                status_code=error.response.status_code,
            ) from error
        except httpx.RequestError as error:
            raise MarieClientError(
                f'Gateway request failed: {type(error).__name__}'
            ) from error

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Track a submitted DAG UUID, or a child job UUID when no DAG exists."""
        try:
            job_id = str(UUID(job_id))
        except (ValueError, TypeError, AttributeError) as error:
            raise MarieClientError('job_id must be a UUID') from error
        try:
            return await self._get(f'/api/operations/dags/{job_id}')
        except MarieClientError as error:
            if error.status_code != 404:
                raise
        return await self._get(f'/api/operations/jobs/{job_id}')

    async def list_jobs(self, state: str | None = None) -> dict[str, Any]:
        """List jobs, optionally filtered by scheduler state."""
        path = '/api/jobs'
        if state:
            path += '/' + quote(state, safe='')
        return await self._get(path)

    async def stop_job(self, job_id: str) -> dict[str, Any]:
        """Reject unsupported gateway cancellation."""
        raise MarieClientError('Job cancellation is not supported by the gateway')

    async def delete_job(self, job_id: str) -> dict[str, Any]:
        """Reject unsupported gateway deletion."""
        raise MarieClientError('Job deletion is not supported by the gateway')

    async def get_deployments(self) -> dict[str, Any]:
        """Get deployment information."""
        return await self._get('/api/deployments')

    async def get_capacity(self) -> dict[str, Any]:
        """Get processing capacity information."""
        return await self._get('/api/capacity')

    async def get_debug_info(
        self, fabric_group_id: str | None = None
    ) -> dict[str, Any]:
        """Get diagnostics for an authorized fabric group."""
        fabric_group_id = fabric_group_id or Config.MARIE_FABRIC_GROUP_ID
        if (
            not fabric_group_id
            or not fabric_group_id.strip()
            or len(fabric_group_id) > 128
        ):
            raise MarieClientError(
                'A fabric_group_id of 1 to 128 characters is required for debug information'
            )
        return await self._get('/api/debug', {'fabric_group_id': fabric_group_id})

    async def health_check(self) -> dict[str, Any]:
        """Check gateway health."""
        return await self._get('/check', {'text': 'health'})

    async def close(self) -> None:
        """Close the HTTP client in its owning event loop."""
        await self._client.aclose()

    async def __aenter__(self) -> 'MarieClient':
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
