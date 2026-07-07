import logging

import pytest

from marie.executor.rag.search_result_doc import SearchResultDoc
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers import marie_gateway as marie_gateway_module
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.marie_gateway import (
    ALLOWED_EXECUTOR_OPS,
    MarieServerGateway,
)
from marie.types_core.request.data import DataRequest


class _FakeResponse:
    def __init__(self, parameters, docs):
        self.parameters = parameters
        self.docs = docs


class _FakeStreamer:
    def __init__(self, result_parameters, result_docs=None):
        self.result_parameters = result_parameters
        self.result_docs = result_docs
        self.received_request = None

    async def process_single_data(self, request, *args, **kwargs):
        self.received_request = request
        if self.result_docs is not None:
            return _FakeResponse(self.result_parameters, self.result_docs)
        response = DataRequest()
        response.parameters = self.result_parameters
        return response


def _gateway(streamer=None):
    gateway = object.__new__(MarieServerGateway)
    gateway.logger = logging.getLogger("test-executor-command")
    gateway.streamer = streamer
    return gateway


async def _drain(gen):
    async for item in gen:
        return item
    raise AssertionError("handle_executor_command yielded nothing")


def test_allow_list_is_exactly_the_spec():
    assert ALLOWED_EXECUTOR_OPS == {
        "search": "/search",
        "hybrid_search": "/hybrid_search",
        "index_stats": "/index_stats",
        "source_stats": "/source_stats",
        "delete_source": "/delete_source",
        "delete_index": "/delete_index",
    }


@pytest.mark.asyncio
async def test_unknown_action_rejected():
    gateway = _gateway()
    resp = await _drain(
        gateway.handle_executor_command({"action": "drop_all", "metadata": {}})
    )
    assert resp.parameters["status"] == "error"
    assert "drop_all" in resp.parameters["msg"]


@pytest.mark.asyncio
async def test_hybrid_search_routes_to_vector_store_executor():
    fake_streamer = _FakeStreamer(result_parameters={"hits": []})
    gateway = _gateway(streamer=fake_streamer)

    metadata = {"index_name": "kb-1", "query": "invoice"}
    resp = await _drain(
        gateway.handle_executor_command(
            {"action": "hybrid_search", "metadata": metadata}
        )
    )

    sent = fake_streamer.received_request
    assert sent.header.exec_endpoint == "/hybrid_search"
    assert sent.header.target_executor == "vector_store_executor"
    assert dict(sent.parameters) == metadata
    assert resp.parameters["status"] == "ok"
    assert resp.parameters["result"] == {"hits": []}
    assert resp.parameters["docs"] == []


@pytest.mark.asyncio
async def test_search_serializes_docs_into_response():
    # Mirrors what VectorStoreExecutor.search returns: a real SearchResultDoc
    # (declared pydantic fields, not a bag of metadata) with no text/rrf score.
    result_doc = SearchResultDoc(
        id="node-1",
        text="hello world",
        similarity=0.87,
        source_id="src-1",
        node_type="chunk",
        index_name="kb-1",
        ref_doc_id="ref-1",
    )
    fake_streamer = _FakeStreamer(result_parameters={}, result_docs=[result_doc])
    gateway = _gateway(streamer=fake_streamer)

    resp = await _drain(
        gateway.handle_executor_command(
            {"action": "search", "metadata": {"query": "invoice", "top_k": 5}}
        )
    )

    assert resp.parameters["status"] == "ok"
    docs = resp.parameters["docs"]
    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == "node-1"
    assert doc["text"] == "hello world"
    assert doc["score"] == 0.87
    assert doc["source_id"] == "src-1"
    assert doc["node_type"] == "chunk"
    assert doc["index_name"] == "kb-1"
    assert doc["ref_doc_id"] == "ref-1"
    assert "text_score" not in doc
    assert "rrf_score" not in doc


@pytest.mark.asyncio
async def test_hybrid_search_docs_include_text_and_rrf_scores():
    # Hybrid search populates text_score/rrf_score; the fused rrf_score
    # should win as the top-level "score", and both should be surfaced.
    result_doc = SearchResultDoc(
        id="node-2",
        text="fused result",
        similarity=0.6,
        text_score=0.3,
        rrf_score=0.05,
        source_id="src-2",
        node_type="chunk",
        index_name="kb-1",
        ref_doc_id="ref-2",
    )
    fake_streamer = _FakeStreamer(result_parameters={}, result_docs=[result_doc])
    gateway = _gateway(streamer=fake_streamer)

    resp = await _drain(
        gateway.handle_executor_command(
            {"action": "hybrid_search", "metadata": {"query": "invoice"}}
        )
    )

    doc = resp.parameters["docs"][0]
    assert doc["score"] == 0.05
    assert doc["text_score"] == 0.3
    assert doc["rrf_score"] == 0.05


@pytest.mark.asyncio
async def test_dict_op_response_is_unchanged_and_has_empty_docs():
    fake_streamer = _FakeStreamer(
        result_parameters={"index_name": "kb-1", "node_count": 42}
    )
    gateway = _gateway(streamer=fake_streamer)

    resp = await _drain(
        gateway.handle_executor_command(
            {"action": "index_stats", "metadata": {"index_name": "kb-1"}}
        )
    )

    assert resp.parameters["status"] == "ok"
    assert resp.parameters["result"] == {"index_name": "kb-1", "node_count": 42}
    assert resp.parameters["docs"] == []


def _patch_setup_server_collaborators(monkeypatch, sensor_calls):
    """Stub out everything setup_server() touches besides the sensor wiring."""

    monkeypatch.setattr(marie_gateway_module, "setup_toast_events", lambda cfg: None)
    monkeypatch.setattr(marie_gateway_module, "setup_storage", lambda cfg: None)
    monkeypatch.setattr(marie_gateway_module, "setup_auth", lambda cfg: None)
    monkeypatch.setattr(
        marie_gateway_module, "setup_llm_tracking", lambda *args, **kwargs: None
    )

    def fake_setup_sensor_worker(sensor_config, db_config):
        sensor_calls.append((sensor_config, db_config))

    monkeypatch.setattr(
        marie_gateway_module, "setup_sensor_worker", fake_setup_sensor_worker
    )

    async def fake_super_setup_server(self):
        return None

    monkeypatch.setattr(CompositeServer, "setup_server", fake_super_setup_server)

    async def fake_setup_service_discovery(self, **kwargs):
        return None

    monkeypatch.setattr(
        MarieServerGateway, "setup_service_discovery", fake_setup_service_discovery
    )


def _gateway_for_setup_server(args):
    gateway = object.__new__(MarieServerGateway)
    gateway.logger = logging.getLogger("test-setup-server")
    gateway.servers = []
    gateway.args = args
    return gateway


@pytest.mark.asyncio
async def test_setup_server_starts_sensor_worker_when_configured(monkeypatch):
    sensor_calls = []
    _patch_setup_server_collaborators(monkeypatch, sensor_calls)

    sensor_config = {"enabled": True, "daemon_interval_seconds": 5}
    kv_store_kwargs = {"provider": "postgresql", "hostname": "localhost"}
    gateway = _gateway_for_setup_server(
        {
            "sensors": sensor_config,
            "kv_store_kwargs": kv_store_kwargs,
            "discovery_host": "0.0.0.0",
            "discovery_port": 2379,
            "discovery_service_name": "gateway/marie",
        }
    )

    await gateway.setup_server()

    assert sensor_calls == [(sensor_config, kv_store_kwargs)]


@pytest.mark.asyncio
async def test_setup_server_sensor_worker_noop_when_sensors_config_absent(
    monkeypatch,
):
    sensor_calls = []
    _patch_setup_server_collaborators(monkeypatch, sensor_calls)

    # No "sensors" key at all -- must not raise, and setup_sensor_worker
    # itself no-ops on an empty config.
    gateway = _gateway_for_setup_server(
        {
            "kv_store_kwargs": {"provider": "postgresql", "hostname": "localhost"},
            "discovery_host": "0.0.0.0",
            "discovery_port": 2379,
            "discovery_service_name": "gateway/marie",
        }
    )

    await gateway.setup_server()

    assert sensor_calls == [({}, {"provider": "postgresql", "hostname": "localhost"})]
