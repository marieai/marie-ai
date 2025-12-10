"""
Tests for S3StateBackend.

These tests use moto to mock S3 operations.
"""

from unittest.mock import MagicMock, patch

import pytest

from marie_kernel import TaskInstanceRef

# Skip all tests if boto3 is not installed
boto3 = pytest.importorskip("boto3")

# Try to import moto for integration-style tests
try:
    from moto import mock_aws

    HAS_MOTO = True
except ImportError:
    HAS_MOTO = False
    mock_aws = None

from marie_kernel.backends.s3 import S3StateBackend


@pytest.fixture
def task_ref():
    """Create a standard TaskInstanceRef for testing."""
    return TaskInstanceRef(
        tenant_id="test_tenant",
        dag_name="test_dag",
        dag_id="run_001",
        task_id="task_1",
        try_number=1,
    )


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = MagicMock()
    return client


@pytest.fixture
def s3_backend(mock_s3_client):
    """Create an S3StateBackend with mock client."""
    return S3StateBackend(mock_s3_client, "test-bucket", "marie-state")


class TestS3BackendKeyBuilding:
    """Tests for S3 key path construction."""

    def test_build_key_format(self, s3_backend, task_ref):
        """Test that S3 keys follow expected format."""
        key = s3_backend._build_key(task_ref, "MY_KEY")
        expected = "marie-state/test_tenant/test_dag/run_001/task_1/1/MY_KEY.json"
        assert key == expected

    def test_build_key_with_task_override(self, s3_backend, task_ref):
        """Test building key with task_id override."""
        key = s3_backend._build_key(task_ref, "MY_KEY", task_id_override="other_task")
        expected = "marie-state/test_tenant/test_dag/run_001/other_task/1/MY_KEY.json"
        assert key == expected

    def test_build_task_prefix_with_try(self, s3_backend, task_ref):
        """Test building task prefix including try_number."""
        prefix = s3_backend._build_task_prefix(task_ref, include_try_number=True)
        expected = "marie-state/test_tenant/test_dag/run_001/task_1/1/"
        assert prefix == expected

    def test_build_task_prefix_without_try(self, s3_backend, task_ref):
        """Test building task prefix without try_number."""
        prefix = s3_backend._build_task_prefix(task_ref, include_try_number=False)
        expected = "marie-state/test_tenant/test_dag/run_001/task_1/"
        assert prefix == expected


class TestS3BackendPush:
    """Tests for push operation."""

    def test_push_calls_put_object(self, s3_backend, mock_s3_client, task_ref):
        """Test that push calls S3 put_object correctly."""
        s3_backend.push(task_ref, "MY_KEY", {"data": 123})

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]

        assert call_kwargs["Bucket"] == "test-bucket"
        assert (
            call_kwargs["Key"]
            == "marie-state/test_tenant/test_dag/run_001/task_1/1/MY_KEY.json"
        )
        assert call_kwargs["ContentType"] == "application/json"

        # Verify body content
        import json

        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert body["value"] == {"data": 123}
        assert body["metadata"] == {}
        assert body["task_instance"]["tenant_id"] == "test_tenant"

    def test_push_with_metadata(self, s3_backend, mock_s3_client, task_ref):
        """Test that metadata is included in the payload."""
        s3_backend.push(
            task_ref, "KEY", "value", metadata={"source": "test", "version": 1}
        )

        import json

        call_kwargs = mock_s3_client.put_object.call_args[1]
        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert body["metadata"] == {"source": "test", "version": 1}


class TestS3BackendPull:
    """Tests for pull operation."""

    def test_pull_returns_value(self, s3_backend, mock_s3_client, task_ref):
        """Test that pull returns the stored value."""
        import json

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"value": {"data": 42}}).encode()
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        result = s3_backend.pull(task_ref, "MY_KEY")

        assert result == {"data": 42}
        mock_s3_client.get_object.assert_called_once()

    def test_pull_returns_default_on_not_found(
        self, s3_backend, mock_s3_client, task_ref
    ):
        """Test that pull returns default when key not found."""
        from botocore.exceptions import ClientError

        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        result = s3_backend.pull(task_ref, "MISSING", default="fallback")

        assert result == "fallback"

    def test_pull_searches_from_tasks_in_order(
        self, s3_backend, mock_s3_client, task_ref
    ):
        """Test that from_tasks are searched in order."""
        import json

        from botocore.exceptions import ClientError

        # First task not found, second task found
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"value": "from_task_b"}).encode()

        def get_object_side_effect(**kwargs):
            if "task_a" in kwargs["Key"]:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": mock_body}

        mock_s3_client.get_object.side_effect = get_object_side_effect

        result = s3_backend.pull(task_ref, "KEY", from_tasks=["task_a", "task_b"])

        assert result == "from_task_b"
        assert mock_s3_client.get_object.call_count == 2


class TestS3BackendClearForTask:
    """Tests for clear_for_task operation."""

    def test_clear_lists_and_deletes_objects(
        self, s3_backend, mock_s3_client, task_ref
    ):
        """Test that clear_for_task deletes all objects for task."""
        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "marie-state/test_tenant/test_dag/run_001/task_1/1/KEY_1.json"
                    },
                    {
                        "Key": "marie-state/test_tenant/test_dag/run_001/task_1/1/KEY_2.json"
                    },
                    {
                        "Key": "marie-state/test_tenant/test_dag/run_001/task_1/2/KEY_1.json"
                    },
                ]
            }
        ]

        s3_backend.clear_for_task(task_ref)

        # Verify delete was called
        mock_s3_client.delete_objects.assert_called_once()
        call_kwargs = mock_s3_client.delete_objects.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert len(call_kwargs["Delete"]["Objects"]) == 3

    def test_clear_handles_empty_list(self, s3_backend, mock_s3_client, task_ref):
        """Test that clear handles no objects gracefully."""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]

        s3_backend.clear_for_task(task_ref)

        # delete_objects should not be called when there's nothing to delete
        mock_s3_client.delete_objects.assert_not_called()


class TestS3BackendExists:
    """Tests for exists helper method."""

    def test_exists_returns_true(self, s3_backend, mock_s3_client, task_ref):
        """Test exists returns True when object exists."""
        mock_s3_client.head_object.return_value = {}

        result = s3_backend.exists(task_ref, "MY_KEY")

        assert result is True
        mock_s3_client.head_object.assert_called_once()

    def test_exists_returns_false_on_404(self, s3_backend, mock_s3_client, task_ref):
        """Test exists returns False when object not found."""
        from botocore.exceptions import ClientError

        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )

        result = s3_backend.exists(task_ref, "MISSING")

        assert result is False


class TestS3BackendGetAllForTask:
    """Tests for get_all_for_task helper."""

    def test_get_all_returns_dict(self, s3_backend, mock_s3_client, task_ref):
        """Test get_all_for_task returns all keys."""
        import json

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "marie-state/test_tenant/test_dag/run_001/task_1/1/KEY_1.json"
                    },
                    {
                        "Key": "marie-state/test_tenant/test_dag/run_001/task_1/1/KEY_2.json"
                    },
                ]
            }
        ]

        def get_object_side_effect(**kwargs):
            mock_body = MagicMock()
            if "KEY_1" in kwargs["Key"]:
                mock_body.read.return_value = json.dumps({"value": "value_1"}).encode()
            else:
                mock_body.read.return_value = json.dumps({"value": "value_2"}).encode()
            return {"Body": mock_body}

        mock_s3_client.get_object.side_effect = get_object_side_effect

        result = s3_backend.get_all_for_task(task_ref)

        assert result == {"KEY_1": "value_1", "KEY_2": "value_2"}


class TestS3BackendProtocolCompliance:
    """Tests verifying StateBackend protocol compliance."""

    def test_implements_push(self, s3_backend, task_ref):
        """Test that push method exists with correct signature."""
        # Should not raise
        s3_backend.push(task_ref, "key", "value", metadata={"a": 1})

    def test_implements_pull(self, s3_backend, mock_s3_client, task_ref):
        """Test that pull method exists with correct signature."""
        from botocore.exceptions import ClientError

        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        # Should not raise
        s3_backend.pull(task_ref, "key", from_tasks=["a", "b"], default="x")

    def test_implements_clear_for_task(self, s3_backend, mock_s3_client, task_ref):
        """Test that clear_for_task method exists."""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]
        # Should not raise
        s3_backend.clear_for_task(task_ref)


@pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
class TestS3BackendWithMoto:
    """Integration tests using moto to mock AWS S3."""

    @pytest.fixture
    def moto_s3_backend(self):
        """Create S3 backend with moto-mocked S3."""
        with mock_aws():
            s3_client = boto3.client("s3", region_name="us-east-1")
            s3_client.create_bucket(Bucket="test-bucket")
            backend = S3StateBackend(s3_client, "test-bucket", "marie-state")
            yield backend

    def test_push_and_pull_roundtrip(self, moto_s3_backend, task_ref):
        """Test full push/pull roundtrip with moto."""
        moto_s3_backend.push(task_ref, "MY_KEY", {"nested": {"data": [1, 2, 3]}})
        result = moto_s3_backend.pull(task_ref, "MY_KEY")
        assert result == {"nested": {"data": [1, 2, 3]}}

    def test_pull_from_upstream_task(self, moto_s3_backend):
        """Test pulling from upstream task with moto."""
        upstream_ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="upstream",
            try_number=1,
        )
        downstream_ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="downstream",
            try_number=1,
        )

        moto_s3_backend.push(upstream_ti, "RESULT", "upstream_value")
        result = moto_s3_backend.pull(downstream_ti, "RESULT", from_tasks=["upstream"])
        assert result == "upstream_value"

    def test_clear_for_task_with_moto(self, moto_s3_backend, task_ref):
        """Test clear_for_task with moto."""
        # Push multiple keys
        moto_s3_backend.push(task_ref, "KEY_1", "value_1")
        moto_s3_backend.push(task_ref, "KEY_2", "value_2")

        # Also push with different try_number
        ti2 = task_ref.with_try_number(2)
        moto_s3_backend.push(ti2, "KEY_1", "try2_value")

        # Clear should remove all
        moto_s3_backend.clear_for_task(task_ref)

        # All keys should be gone
        assert moto_s3_backend.pull(task_ref, "KEY_1") is None
        assert moto_s3_backend.pull(task_ref, "KEY_2") is None
        assert moto_s3_backend.pull(ti2, "KEY_1") is None

    def test_get_all_for_task_with_moto(self, moto_s3_backend, task_ref):
        """Test get_all_for_task with moto."""
        moto_s3_backend.push(task_ref, "KEY_1", "value_1")
        moto_s3_backend.push(task_ref, "KEY_2", {"nested": True})

        result = moto_s3_backend.get_all_for_task(task_ref)
        assert result == {"KEY_1": "value_1", "KEY_2": {"nested": True}}

    def test_exists_with_moto(self, moto_s3_backend, task_ref):
        """Test exists method with moto."""
        assert moto_s3_backend.exists(task_ref, "MY_KEY") is False

        moto_s3_backend.push(task_ref, "MY_KEY", "value")
        assert moto_s3_backend.exists(task_ref, "MY_KEY") is True
