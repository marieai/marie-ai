"""
Helm render tests for the sandbox seed Jobs (Slices 1–2).

Uses ``helm template`` via subprocess to render the umbrella chart and asserts
that the Wave-1/2/3 seed Jobs:
  - Are present with correct names when sandbox.enabled=true + seed flags set.
  - Carry the expected argocd.argoproj.io/sync-wave annotations.
  - Are ordered (wave 1 < 2 < 3).
  - Are absent when sandbox.enabled=false.

Requires helm ≥ 3.x on PATH.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Absolute path to the umbrella chart directory.
# __file__ is tests/unit/sandbox/test_helm_render.py → parents[3] = project root.
_CHART_DIR = str(
    Path(__file__).resolve().parents[3]
    / 'deploy'
    / 'helm'
    / 'charts'
    / 'marie'
)


def _render(extra_sets: list[str] | None = None) -> list[dict[str, Any]]:
    """Run ``helm template`` and return parsed YAML documents."""
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


def _job_by_name(docs: list[dict[str, Any]], name_suffix: str) -> dict[str, Any] | None:
    for j in _jobs(docs):
        if j.get('metadata', {}).get('name', '').endswith(name_suffix):
            return j
    return None


def _sync_wave(job: dict[str, Any]) -> int:
    annotations = job.get('metadata', {}).get('annotations', {})
    wave_str = annotations.get('argocd.argoproj.io/sync-wave', '')
    return int(wave_str) if wave_str else -1


def _hook_weight(job: dict[str, Any]) -> int:
    annotations = job.get('metadata', {}).get('annotations', {})
    weight_str = annotations.get('helm.sh/hook-weight', '')
    return int(weight_str) if weight_str else -1


# ============================================================== test cases ===


class TestWave1DefaultsJob:
    """Wave-1 seed-defaults Job appears when sandbox.enabled=true."""

    def test_job_present_when_sandbox_enabled(self):
        docs = _render()
        job = _job_by_name(docs, '-sandbox-seed-defaults')
        assert job is not None, 'Wave-1 seed-defaults Job must be rendered'

    def test_sync_wave_is_1(self):
        docs = _render()
        job = _job_by_name(docs, '-sandbox-seed-defaults')
        assert _sync_wave(job) == 1

    def test_hook_weight_is_1(self):
        docs = _render()
        job = _job_by_name(docs, '-sandbox-seed-defaults')
        assert _hook_weight(job) == 1

    def test_command_is_seed(self):
        docs = _render()
        job = _job_by_name(docs, '-sandbox-seed-defaults')
        containers = job['spec']['template']['spec']['containers']
        cmd = containers[0]['command']
        assert 'seed' in cmd

    def test_absent_when_sandbox_disabled(self):
        cmd = [
            'helm',
            'template',
            'sbx-test',
            _CHART_DIR,
            '--set',
            'sandbox.enabled=false',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            pytest.fail(f'helm template failed:\n{result.stderr}')
        docs = list(yaml.safe_load_all(result.stdout))
        docs = [d for d in docs if d is not None]
        job = _job_by_name(docs, '-sandbox-seed-defaults')
        assert job is None, 'Wave-1 Job must NOT be rendered when sandbox.enabled=false'


class TestWave2BlueprintJob:
    """Wave-2 seed-blueprint Job gated on blueprint.enabled=true."""

    def test_absent_when_blueprint_disabled(self):
        docs = _render()  # blueprint.enabled defaults to false
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        assert job is None, 'Wave-2 Job must be absent when blueprint.enabled=false'

    def test_present_when_blueprint_enabled(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        assert job is not None, 'Wave-2 seed-blueprint Job must be rendered'

    def test_sync_wave_is_2(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        assert _sync_wave(job) == 2

    def test_hook_weight_is_2(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        assert _hook_weight(job) == 2

    def test_command_is_install_blueprint(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        containers = job['spec']['template']['spec']['containers']
        cmd = containers[0]['command']
        assert 'install-blueprint' in cmd

    def test_gateway_url_env_var_set(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        containers = job['spec']['template']['spec']['containers']
        env = {e['name']: e for e in containers[0]['env']}
        assert 'SANDBOX_GATEWAY_URL' in env
        # Should reference the in-namespace server service
        gw = env['SANDBOX_GATEWAY_URL']['value']
        assert 'sbx-test-server' in gw

    def test_absent_when_sandbox_disabled(self):
        cmd = [
            'helm',
            'template',
            'sbx-test',
            _CHART_DIR,
            '--set',
            'sandbox.enabled=false',
            '--set',
            'sandbox.seed.blueprint.enabled=true',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        docs = list(yaml.safe_load_all(result.stdout))
        docs = [d for d in docs if d is not None]
        job = _job_by_name(docs, '-sandbox-seed-blueprint')
        assert job is None


class TestWave3PluginsJob:
    """Wave-3 seed-plugins Job gated on plugins.enabled=true."""

    _SETS = [
        'sandbox.seed.plugins.enabled=true',
        'sandbox.seed.plugins.registryUrl=https://plugins.example.com',
        'sandbox.seed.adminApiKeySecret=sbx-test-admin',
    ]

    def test_absent_when_plugins_disabled(self):
        docs = _render()
        job = _job_by_name(docs, '-sandbox-seed-plugins')
        assert job is None

    def test_present_when_plugins_enabled(self):
        docs = _render(self._SETS)
        job = _job_by_name(docs, '-sandbox-seed-plugins')
        assert job is not None

    def test_sync_wave_is_3(self):
        docs = _render(self._SETS)
        job = _job_by_name(docs, '-sandbox-seed-plugins')
        assert _sync_wave(job) == 3

    def test_hook_weight_is_3(self):
        docs = _render(self._SETS)
        job = _job_by_name(docs, '-sandbox-seed-plugins')
        assert _hook_weight(job) == 3

    def test_command_is_install_plugins(self):
        docs = _render(self._SETS)
        job = _job_by_name(docs, '-sandbox-seed-plugins')
        containers = job['spec']['template']['spec']['containers']
        cmd = containers[0]['command']
        assert 'install-plugins' in cmd

    def test_absent_when_sandbox_disabled(self):
        cmd = [
            'helm',
            'template',
            'sbx-test',
            _CHART_DIR,
            '--set',
            'sandbox.enabled=false',
            '--set',
            'sandbox.seed.plugins.enabled=true',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        docs = list(yaml.safe_load_all(result.stdout))
        docs = [d for d in docs if d is not None]
        job = _job_by_name(docs, '-sandbox-seed-plugins')
        assert job is None


class TestWaveOrdering:
    """Verify wave-1 < wave-2 < wave-3 across all three Jobs rendered together."""

    def test_sync_waves_are_strictly_ordered(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.plugins.enabled=true',
            'sandbox.seed.plugins.registryUrl=https://plugins.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        w1 = _sync_wave(_job_by_name(docs, '-sandbox-seed-defaults'))
        w2 = _sync_wave(_job_by_name(docs, '-sandbox-seed-blueprint'))
        w3 = _sync_wave(_job_by_name(docs, '-sandbox-seed-plugins'))
        assert w1 < w2 < w3, f'Expected wave ordering 1<2<3, got {w1}<{w2}<{w3}'

    def test_hook_weights_are_strictly_ordered(self):
        docs = _render([
            'sandbox.seed.blueprint.enabled=true',
            'sandbox.seed.blueprint.id=test-bp',
            'sandbox.seed.blueprint.registryUrl=https://reg.example.com',
            'sandbox.seed.plugins.enabled=true',
            'sandbox.seed.plugins.registryUrl=https://plugins.example.com',
            'sandbox.seed.adminApiKeySecret=sbx-test-admin',
        ])
        h1 = _hook_weight(_job_by_name(docs, '-sandbox-seed-defaults'))
        h2 = _hook_weight(_job_by_name(docs, '-sandbox-seed-blueprint'))
        h3 = _hook_weight(_job_by_name(docs, '-sandbox-seed-plugins'))
        assert h1 < h2 < h3, f'Expected hook weight ordering 1<2<3, got {h1}<{h2}<{h3}'


class TestHelmLint:
    """The chart must pass helm lint after Slice-2 additions."""

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
