#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

python3 - "${REPO_ROOT}" "$@" <<'PY'
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


VERSION_FILE = Path("marie/_version.py")
RELEASE_FILES = (
    VERSION_FILE,
    Path("Dockerfiles/docker-compose.allinone.yml"),
    Path("Dockerfiles/docker-compose.extract.yml"),
    Path("Dockerfiles/docker-compose.g5-annotators.yml"),
    Path("Dockerfiles/docker-compose.gateway.yml"),
    Path("bootstrap-marie.sh"),
    Path("bootstrap.md"),
    Path("build.sh"),
    Path("deploy/README.md"),
    Path("deploy/helm/charts/marie/Chart.yaml"),
    Path("deploy/helm/charts/marie/charts/executor/Chart.yaml"),
    Path("deploy/helm/charts/marie/charts/executor/values.yaml"),
    Path("deploy/helm/charts/marie/charts/server/Chart.yaml"),
    Path("deploy/helm/charts/marie/charts/server/values.yaml"),
    Path("deploy/helm/charts/marie/values-appimage-amd64.yaml"),
    Path("deploy/helm/charts/marie/values-appimage.yaml"),
    Path("deploy/helm/charts/marie/values-local.yaml"),
    Path("deploy/helm/charts/marie/values-production.yaml"),
    Path("deploy/helm/charts/marie/values.yaml"),
    Path("deploy/operator/README.md"),
    Path("deploy/operator/config/samples/marie_v1alpha1_mariecluster.yaml"),
    Path("deploy/operator/deploy/crds/mariecluster-crd.yaml"),
    Path("deploy/smoke-marie-helm.sh"),
    Path("docker-scripts/id"),
    Path("docker-scripts/id.gateway"),
    Path("docker-scripts/run-gateway.sh"),
    Path("docs/deployment/all-in-one.md"),
    Path("docs/docs/getting-started/contributing/build-guide.md"),
    Path("docs/docs/getting-started/deployment/docker.md"),
    Path("packages/marie-cli/tests/test_cli_entrypoint.py"),
    Path("tests/unit/test_build_info.py"),
    Path("vagrant/envs/test-default.env"),
    Path("vagrant/envs/test-full.env"),
)
VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?$"
)
SOURCE_PATTERN = re.compile(
    r'^__version__\s*=\s*["\'](?P<version>[^"\']+)["\']\s*$', re.MULTILINE
)


def usage() -> None:
    print(
        "Usage: scripts/update-version.sh "
        "<VERSION|major|minor|patch|final|rc|dev|--check|--current|--files>\n"
        "       scripts/update-version.sh --resolve "
        "<VERSION|major|minor|patch|final|rc|dev>",
        file=sys.stderr,
    )


def current_version(root: Path) -> str:
    version_source = (root / VERSION_FILE).read_text(encoding="utf-8")
    match = SOURCE_PATTERN.search(version_source)
    if match is None:
        raise RuntimeError(f"Could not read __version__ from {VERSION_FILE}")
    return match.group("version")


def commit_count_since_last_tag(root: Path) -> int:
    last_tag = subprocess.run(
        ["git", "tag", "--sort=-version:refname"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    revision_range = f"{last_tag[0]}..HEAD" if last_tag else "HEAD"
    result = subprocess.run(
        ["git", "rev-list", revision_range, "--count"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def resolve_version(root: Path, current: str, argument: str) -> str:
    if VERSION_PATTERN.fullmatch(argument):
        return argument

    match = re.fullmatch(
        r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
        r"(?P<suffix>(?:(?:a|b|rc)\d+|\.dev\d+)?)",
        current,
    )
    if match is None:
        raise RuntimeError(f"Unsupported current version: {current}")

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    suffix = match.group("suffix")

    if argument == "major":
        return f"{major + 1}.0.0"
    if argument == "minor":
        return f"{major}.{minor + 1}.0"
    if argument == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if argument == "final":
        if suffix:
            return f"{major}.{minor}.{patch}"
        return f"{major}.{minor}.{patch + 1}"
    if argument == "rc":
        rc_match = re.fullmatch(r"rc(?P<number>\d+)", suffix)
        if rc_match:
            return f"{major}.{minor}.{patch}rc{int(rc_match.group('number')) + 1}"
        return f"{major}.{minor}.{patch + 1}rc1"
    if argument == "dev":
        count = commit_count_since_last_tag(root)
        return f"{major}.{minor}.{patch}.dev{count}"

    usage()
    raise SystemExit(2)


def verify_release_files(root: Path, version: str) -> None:
    errors: list[str] = []
    for relative_path in RELEASE_FILES:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing file: {relative_path}")
            continue
        if version not in path.read_text(encoding="utf-8"):
            errors.append(f"missing version {version}: {relative_path}")

    if errors:
        raise RuntimeError("Release version is inconsistent:\n  " + "\n  ".join(errors))


def update_release_files(root: Path, current: str, target: str) -> None:
    verify_release_files(root, current)
    if current == target:
        print(f"Marie release files already use {target}")
        return

    for relative_path in RELEASE_FILES:
        path = root / relative_path
        content = path.read_text(encoding="utf-8")
        path.write_text(content.replace(current, target), encoding="utf-8")

    verify_release_files(root, target)
    print(f"Updated {len(RELEASE_FILES)} release files: {current} -> {target}")


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
    current = current_version(root)

    if arguments == ["--current"]:
        print(current)
        return

    if arguments == ["--files"]:
        for path in RELEASE_FILES:
            print(path)
        return

    if arguments == ["--check"]:
        verify_release_files(root, current)
        print(f"Marie release files consistently use {current}")
        return

    if len(arguments) == 2 and arguments[0] == "--resolve":
        print(resolve_version(root, current, arguments[1]))
        return

    if len(arguments) > 1:
        usage()
        raise SystemExit(2)

    mode = arguments[0] if arguments else "dev"
    target = resolve_version(root, current, mode)
    update_release_files(root, current, target)


if __name__ == "__main__":
    main()
PY
