import os
import subprocess
import sys


def test_flow_module_does_not_reenter_runtime_facade() -> None:
    code = (
        "import sys; "
        "from marie.orchestrate.flow.base import Flow; "
        "assert Flow.__name__ == 'Flow'; "
        "assert 'marie.runtime' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_runtime_accepts_an_already_selected_spawn_context() -> None:
    code = (
        "import multiprocessing as mp; "
        "mp.set_start_method('spawn'); "
        "import marie.runtime; "
        "assert mp.get_start_method() == 'spawn'"
    )
    env = dict(os.environ, JINA_MP_START_METHOD="spawn")

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "failed to set multiprocessing start_method" not in result.stderr
