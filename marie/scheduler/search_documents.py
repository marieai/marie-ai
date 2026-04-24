from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from marie.query_planner.base import QueryPlan
from marie.scheduler.models import WorkInfo

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class JobSearchDocument:
    job_id: str
    queue_name: str
    dag_id: str
    planner: str | None
    job_name: str
    node_label: str | None
    ref_id: str | None
    ref_type: str | None
    asset_uri: str | None
    metadata_queue_id: str | None
    layout: str | None
    mode: str | None
    policy: str | None
    method: str | None
    endpoint: str | None
    executor: str | None
    model_name: str | None
    search_text: str


def build_job_search_documents(
    *,
    plan: QueryPlan,
    dag_nodes: list[WorkInfo],
    planner: str | None,
) -> list[JobSearchDocument]:
    nodes_by_id = {node.task_id: node for node in plan.nodes}
    documents: list[JobSearchDocument] = []

    for work_info in dag_nodes:
        if not work_info.dag_id:
            raise ValueError(f"Missing dag_id for job {work_info.id}")
        node = nodes_by_id.get(work_info.id)
        metadata = _metadata(work_info.data)
        op_params = _mapping(metadata.get("op_params"))
        definition = getattr(node, "definition", None)
        definition_params = _mapping(getattr(definition, "params", None))
        on_value = _text(metadata.get("on"))
        executor = _executor(on_value)

        endpoint = (
            _text(getattr(definition, "endpoint", None))
            or _endpoint_path(on_value)
            or _text(on_value)
        )

        layout = _text(op_params.get("layout")) or _text(
            definition_params.get("layout")
        )

        node_label = _text(metadata.get("name")) or _text(
            getattr(node, "query_str", None)
        )
        method = _text(getattr(definition, "method", None))
        model_name = _text(getattr(definition, "model_name", None)) or _text(
            op_params.get("model_name")
        )

        search_text = _search_text(
            planner,
            work_info.id,
            work_info.dag_id,
            work_info.name,
            node_label,
            metadata.get("ref_id"),
            metadata.get("ref_type"),
            metadata.get("uri"),
            metadata.get("queue_id"),
            layout,
            metadata.get("mode"),
            metadata.get("policy"),
            method,
            endpoint,
            executor,
            model_name,
            on_value,
        )

        documents.append(
            JobSearchDocument(
                job_id=work_info.id,
                queue_name=work_info.name,
                dag_id=work_info.dag_id,
                planner=_text(planner) or _text(metadata.get("planner")),
                job_name=work_info.name,
                node_label=node_label,
                ref_id=_text(metadata.get("ref_id")),
                ref_type=_text(metadata.get("ref_type")),
                asset_uri=_text(metadata.get("uri")),
                metadata_queue_id=_text(metadata.get("queue_id")),
                layout=layout,
                mode=_text(metadata.get("mode")),
                policy=_text(metadata.get("policy")),
                method=method,
                endpoint=endpoint,
                executor=executor,
                model_name=model_name,
                search_text=search_text,
            )
        )

    return documents


def _metadata(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    metadata = data.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _executor(on_value: str | None) -> str | None:
    if not on_value or "://" not in on_value:
        return None
    return _text(on_value.split("://", 1)[0])


def _endpoint_path(on_value: str | None) -> str | None:
    if not on_value or "://" not in on_value:
        return None
    path = on_value.split("://", 1)[1].strip()
    if not path:
        return None
    if path == "noop":
        return path
    return path if path.startswith("/") else f"/{path}"


def _search_text(*values: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = _text(value)
        if not text:
            continue
        normalized = _WHITESPACE.sub(" ", text).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            parts.append(normalized)

    return " ".join(parts)
