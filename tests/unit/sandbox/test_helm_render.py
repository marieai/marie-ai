"""
Helm render tests for the sandbox seed-defaults Job (Slice 2).

Blueprint + plugin seeding is Studio-orchestrated (post-Argo-sync via
Studio's blueprint-import.service), so NO blueprint/plugin seed Jobs should
appear in the chart.

Asserts:
  - Wave-1 seed-defaults Job renders when sandbox.enabled=true.
  - seed-defaults carries the correct argocd.argoproj.io/sync-wave and
    helm.sh/hook-weight annotations.
  - No blueprint or plugin seed Jobs are present in ANY render.
  - All Jobs are absent when sandbox.enabled=false.

Requires helm >= 3.x on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

# Absolute path to the umbrella chart directory.
# __file__ is tests/unit/sandbox/test_helm_render.py -> parents[3] = project root.
_CHART_DIR = str(
    Path(__file__).resolve().parents[3]
    / 'deploy'
    / 'helm'
    / 'charts'
    / 'marie'
)


def _render(extra_sets: list[str] | None = None) -> list[dict[str, Any]]:
    """Run helm template and return parsed YAML documents."""
    cmd = [
        'helm',
        'template',
        'sbx-test',
        _CHART_DIR,
        '-f',
        f'{_CHART_DIR}/values-sandbox.yaml',
        '--set',
        'sandbox.enabled=true',
    ]
    for s in extra_sets or []:
        cmd += ['--set', s]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.fail(f'helm template failed:\nstdout={result.stdout}\nstderr={result.stderr}')

    docs = list(yaml.safe_load_all(result.stdout))
    return [d for d in docs if d is not None]


def _jobs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [d for d in docs if d.get('kind') == 'Job']


def _job_by_name_suffix(docs: list[dict[str, Any]], suffix: str) -> dict[str, Any] | None:
    for j in _jobs(docs):
        if j.get('metadata', {}).get('name', '').endswith(suffix):
            return j
    return None


def _sync_wave(job: dict[str, Any]) -> int:
    return int(
        job.get('metadata', {}).get('annotations', {}).get(
            'argocd.argoproj.io/sync-wave', '-1'
        )
    )


def _hook_weight(job: dict[str, Any]) -> int:
    return int(
        job.get('metadata', {}).get('annotations', {}).get('helm.sh/hook-weight', '-1')
    )


# ============================================================== test cases ===


class TestWave1DefaultsJob:
    """Wave-1 seed-defaults Job is the only seed Job in the chart."""

    def test_present_when_sandbox_enabled(self):
        docs = _render()
        job = _job_by_name_suffix(docs, '-sandbox-seed-defaults')
        assert job is not None, 'sandbox-seed-defaults Job must render when sandbox.enabled=true'

    def test_sync_wave_is_1(self):
        docs = _render()
        job = _job_by_name_suffix(docs, '-sandbox-seed-defaults')
        assert _sync_wave(job) == 1

    def test_hook_weight_is_1(self):
        docs = _render()
        job = _job_by_name_suffix(docs, '-sandbox-seed-defaults')
        assert _hook_weight(job) == 1

    def test_command_invokes_seed(self):
        docs = _render()
        job = _job_by_name_suffix(docs, '-sandbox-seed-defaults')
        cmd = job['spec']['template']['spec']['containers'][0]['command']
        assert 'seed' in cmd

    def test_absent_when_sandbox_disabled(self):
        cmd = [
            'helm', 'template', 'sbx-test', _CHART_DIR,
            '--set', 'sandbox.enabled=false',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            pytest.fail(f'helm template failed:\n{result.stderr}')
        docs = [d for d in yaml.safe_load_all(result.stdout) if d]
        assert _job_by_name_suffix(docs, '-sandbox-seed-defaults') is None


class TestNoBlueprintOrPluginJobs:
    """Blueprint and plugin Jobs must not appear -- seeding is Studio-orchestrated."""

    def test_no_seed_blueprint_job(self):
        docs = _render()
        job = _job_by_name_suffix(docs, '-sandbox-seed-blueprint')
        assert job is None, (
            'sandbox-seed-blueprint Job must NOT render; '
            'blueprint seeding is done by the Studio Sandbox Service post-sync'
        )

    def test_no_seed_plugins_job(self):
        docs = _render()
        job = _job_by_name_suffix(docs, '-sandbox-seed-plugins')
        assert job is None, (
            'sandbox-seed-plugins Job must NOT render; '
            'plugin seeding is done by the Studio Sandbox Service post-sync'
        )

    def test_exactly_one_sandbox_seed_job_total(self):
        """Exactly one sandbox-seed Job (defaults) renders, nothing else."""
        docs = _render()
        sandbox_seed_jobs = [
            j
            for j in _jobs(docs)
            if 'sandbox-seed' in j.get('metadata', {}).get('name', '')
        ]
        names = [j['metadata']['name'] for j in sandbox_seed_jobs]
        assert len(sandbox_seed_jobs) == 1, (
            f'Expected exactly 1 sandbox-seed Job, found {len(sandbox_seed_jobs)}: {names}'
        )


class TestInformationalSeedMetadata:
    """Setting blueprintId / pluginRefs must not break rendering."""

    def test_blueprint_id_renders_without_error(self):
        docs = _render(['sandbox.seed.blueprintId=ner-vlm-ocr-entity-extraction'])
        assert docs

    def test_plugin_refs_render_without_error(self):
        docs = _render([
            r'sandbox.seed.pluginRefs[0].packageId=connector.ocr-engine',
            r'sandbox.seed.pluginRefs[0].version=2.1.0',
        ])
        assert docs

    def test_blueprint_id_does_not_produce_extra_jobs(self):
        docs = _render(['sandbox.seed.blueprintId=some-bp'])
        sandbox_seed_jobs = [
            j
            for j in _jobs(docs)
            if 'sandbox-seed' in j.get('metadata', {}).get('name', '')
        ]
        assert len(sandbox_seed_jobs) == 1


class TestHelmLint:
    """Chart must pass helm lint after Slice-2 rework."""

    def test_helm_lint_passes(self):
        result = subprocess.run(
            ['helm', 'lint', _CHART_DIR],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f'helm lint failed:\n{result.stdout}\n{result.stderr}'
        )
