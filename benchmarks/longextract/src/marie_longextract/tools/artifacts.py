from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from marie.storage import StorageManager


@dataclass(frozen=True)
class LongExtractArtifacts:
    schema_uri: str
    work_uri: str
    output_uri: str

    @property
    def stitched_uri(self) -> str:
        return f"{self.work_uri}stitched-result.json"

    @property
    def findings_uri(self) -> str:
        return f"{self.work_uri}verification-findings.json"


def require_benchmark_metadata(metadata: dict[str, Any]) -> tuple[str, str, str]:
    content_type = metadata.get("content_type")
    if not isinstance(content_type, str) or not content_type:
        raise ValueError("metadata.content_type is required")
    if content_type != "application/pdf":
        raise ValueError(f"Unsupported benchmark content type: {content_type}")

    benchmark = metadata.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("metadata.benchmark is required")
    schema_uri = benchmark.get("schema_uri")
    output_uri = benchmark.get("output_uri")
    if not isinstance(schema_uri, str) or not schema_uri:
        raise ValueError("metadata.benchmark.schema_uri is required")
    if not isinstance(output_uri, str) or not output_uri:
        raise ValueError("metadata.benchmark.output_uri is required")
    work_uri = benchmark.get("work_uri")
    if not isinstance(work_uri, str) or not work_uri:
        work_uri = output_uri.rsplit("/", 1)[0] + "/work/"
    return schema_uri, output_uri, work_uri.rstrip("/") + "/"


def read_json(uri: str, storage: Any = StorageManager) -> dict[str, Any]:
    raw = storage.read(uri)
    data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if not isinstance(data, dict):
        raise ValueError(f"{uri} did not contain a JSON object")
    return data


def write_json(
    uri: str,
    data: dict[str, Any],
    storage: Any = StorageManager,
) -> None:
    storage.write(
        json.dumps(data, indent=2, sort_keys=True).encode("utf-8"),
        uri,
        overwrite=True,
    )
