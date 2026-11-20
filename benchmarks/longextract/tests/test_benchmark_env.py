from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parents[1] / 'tools' / 'run-with-env.py'


@pytest.mark.parametrize('custom_file', [False, True])
def test_benchmark_file_overrides_inherited_endpoint(
    tmp_path: Path, custom_file: bool
) -> None:
    assert RUNNER.is_file(), 'The benchmark environment launcher is missing'
    tools = tmp_path / 'benchmark' / 'tools'
    tools.mkdir(parents=True)
    script = tools / RUNNER.name
    shutil.copy2(RUNNER, script)
    env_file = tools.parent / ('custom testing.env' if custom_file else '.env')
    env_file.write_text(
        'OPENAI_API_BASE="http://testing.example/v1"\n'
        'OPENAI_API_KEY=fixture-key\n'
        'LLM_QUEUE_ENABLED=false\n',
        encoding='utf-8',
    )
    env = dict(os.environ)
    env.update(
        OPENAI_API_BASE='http://old.example/v1',
        OPENAI_API_KEY='inherited-fixture-key',
        LLM_QUEUE_ENABLED='true',
        BENCHMARK_TEST_VALUE='preserved',
    )
    command = [sys.executable, str(script)]
    if custom_file:
        command.extend(['--env-file', str(env_file)])
    command.extend(
        [
            '--',
            sys.executable,
            '-c',
            'import json, os, sys; '
            'print(json.dumps({key: os.environ.get(key) for key in '
            '["OPENAI_API_BASE", "OPENAI_API_KEY", "LLM_QUEUE_ENABLED", '
            '"MARIE_SKIP_LOAD_ENV", "BENCHMARK_TEST_VALUE"]})); '
            'assert sys.argv[1:] == ["argument with spaces"]; sys.exit(23)',
            'argument with spaces',
        ]
    )
    result = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 23, result.stderr
    assert json.loads(result.stdout) == {
        'OPENAI_API_BASE': 'http://testing.example/v1',
        'OPENAI_API_KEY': 'fixture-key',
        'LLM_QUEUE_ENABLED': 'false',
        'MARIE_SKIP_LOAD_ENV': 'true',
        'BENCHMARK_TEST_VALUE': 'preserved',
    }


def test_missing_env_file_stops_before_starting_command(tmp_path: Path) -> None:
    missing = tmp_path / 'missing.env'
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            '--env-file',
            str(missing),
            '--',
            sys.executable,
            '-c',
            'print("child started")',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert str(missing) in result.stderr
    assert 'child started' not in result.stdout


@pytest.mark.parametrize('endpoint_line', ['OPENAI_API_BASE=', 'OPENAI_API_BASE'])
def test_blank_endpoint_does_not_fall_back_to_inherited_endpoint(
    tmp_path: Path,
    endpoint_line: str,
) -> None:
    env_file = tmp_path / 'blank.env'
    env_file.write_text(f'{endpoint_line}\nOPENAI_API_KEY=EMPTY\n', encoding='utf-8')
    env = dict(os.environ, OPENAI_API_BASE='http://old.example/v1')
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            '--env-file',
            str(env_file),
            '--',
            sys.executable,
            '-c',
            'print("child started")',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert 'OPENAI_API_BASE' in result.stderr
    assert 'child started' not in result.stdout
