#!/usr/bin/env python3
"""Exercise LLM timeout, invalid-output, and resume behavior against AIMock."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from marie.engine.openai_engine import OpenAIEngine
from marie.engine.output_parser import JSONOutputParserError
from PIL import Image

from marie.extract.annotators.llm_annotator import LLMAnnotator


def fault_profile(admin_url: str, profile: str, **settings: object) -> dict[str, Any]:
    payload = json.dumps({"profile": profile, **settings}).encode()
    request = urllib.request.Request(
        f"{admin_url.rstrip('/')}/fault-profile",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def make_working_dir(root: Path, name: str, page_count: int) -> Path:
    working_dir = root / name
    frames_dir = working_dir / "frames"
    frames_dir.mkdir(parents=True)
    for page in range(1, page_count + 1):
        Image.new("RGB", (32, 32), color=(page * 20, 40, 60)).save(
            frames_dir / f"{page:05d}.png"
        )
    return working_dir


def make_annotator(
    working_dir: Path, prompt_dir: Path, engine: OpenAIEngine
) -> LLMAnnotator:
    config = {
        "name": "mock-extract",
        "annotator_type": "llm",
        "mode": "per-page",
        "model_config": {
            "model_name": "gpt-4o",
            "prompt_path": "prompt.j2",
            "system_prompt_text": "Return JSON only.",
            "multimodal": False,
            "expect_output": "json",
            "mini_batch_size": 1,
        },
    }
    empty_context_manager = SimpleNamespace(has_providers=lambda: False)
    with (
        patch(
            "marie.extract.annotators.llm_annotator.route_llm_engine",
            return_value=engine,
        ),
        patch(
            "marie.extract.annotators.llm_annotator.ContextProviderManager",
            return_value=empty_context_manager,
        ),
    ):
        return LLMAnnotator(
            str(working_dir),
            config,
            {"layout_id": "mock"},
            prompt_dir=str(prompt_dir),
        )


def assert_json_files(output_dir: Path, names: list[str]) -> None:
    for name in names:
        value = json.loads((output_dir / name).read_text(encoding="utf-8"))
        assert isinstance(value, (dict, list)), f"{name} is not structured JSON"


def assert_request_count(state: dict[str, Any], expected: int, phase: str) -> None:
    actual = state.get("requestCount")
    assert actual == expected, f"{phase}: expected {expected} requests, got {actual}"


def run(args: argparse.Namespace) -> dict[str, object]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to the AIMock test key")

    with tempfile.TemporaryDirectory(prefix="marie-aimock-e2e-") as temp_dir:
        root = Path(temp_dir)
        (root / "prompt.j2").write_text(
            "extract document fields as JSON\n{{ OCR_DATA }}\n",
            encoding="utf-8",
        )
        engine = OpenAIEngine(
            model_name="gpt-4o",
            base_url=args.base_url,
            is_multimodal=False,
            max_concurrency=2,
            batch_timeout=args.normal_timeout,
            queue_enabled=False,
        )
        prompt_lines = {
            page: [{"text": f"page {page + 1} mock content"}] for page in range(3)
        }
        document = SimpleNamespace(source_metadata={"pages": 3, "ocr": []})

        try:
            with patch(
                "marie.extract.annotators.util._prompt_lines_by_page",
                return_value=prompt_lines,
            ):
                working_dir = make_working_dir(root, "resume", 3)
                annotator = make_annotator(working_dir, root, engine)
                output_dir = Path(annotator.output_dir)

                fault_profile(args.admin_url, "normal", resetCounters=True)
                annotator.annotate(document, [])
                assert_json_files(
                    output_dir, ["00001.json", "00002.json", "00003.json"]
                )
                marker = output_dir / "_SUCCESS.yaml"
                assert marker.is_file()
                assert_request_count(
                    fault_profile(args.admin_url, "normal"), 3, "initial run"
                )

                first_result = output_dir / "00001.json"
                first_mtime = first_result.stat().st_mtime_ns
                marker.unlink()
                (output_dir / "00002.json").write_text("{broken", encoding="utf-8")
                (output_dir / "00003.json").unlink()

                fault_profile(args.admin_url, "invalid_json", resetCounters=True)
                try:
                    annotator.annotate(document, [])
                except JSONOutputParserError:
                    pass
                else:
                    raise AssertionError("invalid JSON run unexpectedly succeeded")

                assert first_result.stat().st_mtime_ns == first_mtime
                assert not marker.exists()
                assert not (output_dir / "00003.json").exists()
                assert not list(output_dir.glob("*.tmp"))
                assert_request_count(
                    fault_profile(args.admin_url, "invalid_json"),
                    2,
                    "invalid JSON run",
                )

                fault_profile(args.admin_url, "normal", resetCounters=True)
                annotator.annotate(document, [])
                assert_json_files(
                    output_dir, ["00001.json", "00002.json", "00003.json"]
                )
                assert first_result.stat().st_mtime_ns == first_mtime
                assert marker.is_file()
                assert_request_count(
                    fault_profile(args.admin_url, "normal"), 2, "resume run"
                )

                timeout_dir = make_working_dir(root, "timeout", 2)
                timeout_annotator = make_annotator(timeout_dir, root, engine)
                timeout_output = Path(timeout_annotator.output_dir)
                engine.batch_processor.batch_timeout = args.failure_timeout
                fault_profile(
                    args.admin_url,
                    "timeout",
                    timeoutMs=int(args.mock_delay * 1000),
                    resetCounters=True,
                )
                started = time.monotonic()
                try:
                    timeout_annotator.annotate(document, [])
                except TimeoutError:
                    pass
                else:
                    raise AssertionError("timeout run unexpectedly succeeded")
                timeout_elapsed = time.monotonic() - started
                assert timeout_elapsed < args.mock_delay
                assert not list(timeout_output.glob("*.json"))
                assert not (timeout_output / "_SUCCESS.yaml").exists()
                assert_request_count(
                    fault_profile(args.admin_url, "timeout"), 2, "timeout run"
                )

                engine.batch_processor.batch_timeout = args.normal_timeout
                fault_profile(args.admin_url, "normal", resetCounters=True)
                timeout_annotator.annotate(document, [])
                assert_json_files(timeout_output, ["00001.json", "00002.json"])
                assert (timeout_output / "_SUCCESS.yaml").is_file()
                assert_request_count(
                    fault_profile(args.admin_url, "normal"), 2, "timeout retry"
                )

            return {
                "invalid_json_resume": "passed",
                "timeout_recovery": "passed",
                "timeout_elapsed_seconds": round(timeout_elapsed, 3),
                "valid_output_preserved": True,
                "duplicate_requests": 0,
            }
        finally:
            fault_profile(args.admin_url, "normal")
            engine.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:4010/v1",
        help="AIMock OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--admin-url",
        default="http://127.0.0.1:4011",
        help="AIMock admin server URL",
    )
    parser.add_argument("--normal-timeout", type=float, default=5.0)
    parser.add_argument("--failure-timeout", type=float, default=0.25)
    parser.add_argument("--mock-delay", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
