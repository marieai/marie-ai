from pathlib import Path

from marie.plugin_daemon import discover_daemon


def test_installed_daemon_discovery(tmp_path: Path) -> None:
    binary = tmp_path / "bin" / "marie-plugin-daemon"
    binary.parent.mkdir()
    binary.touch(mode=0o755)
    env = {"MARIE_DEFAULT_MOUNT": str(tmp_path), "PATH": ""}
    found = discover_daemon(None, None, "localhost:8099", env=env)
    assert found.source == "installed_binary"
    assert found.binary == str(binary)
    assert discover_daemon("http://external:8099", None, "localhost:8099", env=env).source == "url"
