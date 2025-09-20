import time
from unittest.mock import Mock, patch

import pytest
from docarray import DocList
from docarray.documents import TextDoc
from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.constants import DEPLOYMENT_STATUS_PREFIX
from marie.job.job_supervisor import JobSupervisor
from marie.types_core.request.data import DataRequest


@pytest.fixture
def mock_supervisor_dependencies():
    """Fixture to provide mocked dependencies for JobSupervisor."""
    etcd_client_mock = Mock()
    # Mock the lease cache that is now expected to be on the supervisor
    with patch('marie.job.job_supervisor.LeaseCache') as mock_lease_cache_class:
        mock_lease_cache_instance = mock_lease_cache_class.return_value
        return {
            "job_id": "test-job-id",
            "job_info_client": Mock(),
            "job_distributor": Mock(),
            "event_publisher": Mock(),
            "etcd_client": etcd_client_mock,
            "confirmation_event": Mock(),
            "_lease_cache": mock_lease_cache_instance
        }


def test_send_callback_sync(mock_supervisor_dependencies, caplog):
    """Test the functionality of _send_callback_sync with the lease cache."""
    # Arrange
    # We need to manually add the mocked lease cache to the supervisor instance
    lease_cache_mock = mock_supervisor_dependencies.pop("_lease_cache")
    supervisor = JobSupervisor(**mock_supervisor_dependencies)
    supervisor._lease_cache = lease_cache_mock

    # Ensure there is no event loop to test the thread-safe path directly
    supervisor._loop = None

    request = DataRequest()
    request.document_array_cls = DocList[TextDoc]
    request.data.docs = DocList[TextDoc]([TextDoc(text="test")])

    request_info = {
        "request_id": "test-request-id",
        "address": "127.0.0.1:12345",
        "deployment": "test-deployment",
    }

    mock_lease = Mock()
    lease_cache_mock.get_or_refresh.return_value = mock_lease

    # Mock time.monotonic to control time in the test
    with patch("time.monotonic") as mock_time:
        # Arrange a sequence of time values for monotonic()
        mock_time.side_effect = [100.0, 100.1, 100.2, 100.5, 100.8]

        # Act
        supervisor._send_callback_sync([request], request_info)

        # Assert
        # 1. Confirmation event is signaled
        mock_supervisor_dependencies["confirmation_event"].set.assert_called_once()

        # 2. Etcd lease and put are called correctly
        lease_cache_mock.get_or_refresh.assert_called_once_with(
            f"deployments/status/127.0.0.1:12345/test-deployment", ttl=5
        )

        status = HealthCheckResponse.ServingStatus.SERVING
        status_str = HealthCheckResponse.ServingStatus.Name(status)
        expected_key = f"{DEPLOYMENT_STATUS_PREFIX}/{request_info['address']}/{request_info['deployment']}"
        mock_supervisor_dependencies["etcd_client"].put.assert_called_once_with(expected_key, status_str,
                                                                                lease=mock_lease)

        # 3. Log messages are correct
        assert "Sent request to 127.0.0.1:12345 on deployment test-deployment" in caplog.text
        # lease time = 100.2 - 100.1 = 0.1
        # put time = 100.5 - 100.2 = 0.3
        # total = 100.5 - 100.1 = 0.4
        assert "Etcd update for deployments/status/127.0.0.1:12345/test-deployment: lease=0.100s put=0.300s total=0.400s" in caplog.text


def test_send_callback_sync_empty_requests(mock_supervisor_dependencies, caplog):
    """Test _send_callback_sync with empty requests list."""
    # Arrange
    supervisor = JobSupervisor(**mock_supervisor_dependencies)
    request_info = {
        "request_id": "test-request-id",
        "address": "127.0.0.1:12345",
        "deployment": "test-deployment",
    }

    # Act
    supervisor._send_callback_sync([], request_info)

    # Assert
    assert "No valid requests provided." in caplog.text
    mock_supervisor_dependencies["etcd_client"].put.assert_not_called()


def test_send_callback_sync_missing_request_info(mock_supervisor_dependencies, caplog):
    """Test _send_callback_sync with missing keys in request_info."""
    # Arrange
    supervisor = JobSupervisor(**mock_supervisor_dependencies)
    request = DataRequest()

    # Act
    supervisor._send_callback_sync([request], {"request_id": "test"})

    # Assert
    assert "Missing required keys in request_info" in caplog.text
    mock_supervisor_dependencies["etcd_client"].put.assert_not_called()