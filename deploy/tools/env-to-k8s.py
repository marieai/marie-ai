#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

SECRET_RE = re.compile(r"(PASSWORD|SECRET|TOKEN|KEY|CREDENTIAL)", re.IGNORECASE)
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""

    if value[0] in {"'", '"'}:
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid quoted value: {exc}") from exc
        return parts[0] if parts else ""

    return re.sub(r"\s+#.*$", "", value).strip()


def parse_env(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    warnings: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()

        if "=" not in stripped:
            warnings.append(f"line {line_no}: skipped line without '='")
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not KEY_RE.match(key):
            warnings.append(f"line {line_no}: skipped invalid key {key!r}")
            continue

        try:
            value = parse_value(raw_value)
        except ValueError as exc:
            warnings.append(f"line {line_no}: skipped {key}: {exc}")
            continue

        if key in values:
            warnings.append(f"line {line_no}: duplicate key {key}; using last value")
        values[key] = value

    return values, warnings


def split_values(values: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    config: dict[str, str] = {}
    secrets: dict[str, str] = {}

    for key, value in values.items():
        target = secrets if SECRET_RE.search(key) else config
        target[key] = value

    return config, secrets


def metadata(kind: str, name: str, namespace: str) -> list[str]:
    return [
        "apiVersion: v1",
        f"kind: {kind}",
        "metadata:",
        f"  name: {name}",
        f"  namespace: {namespace}",
    ]


def next_backup_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.{timestamp}.bak")
    index = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{timestamp}.{index}.bak")
        index += 1
    return backup


def write_with_backup(path: Path, content: str) -> tuple[str, Path | None]:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return "unchanged", None

        backup = next_backup_path(path)
        backup.write_text(existing, encoding="utf-8")
        path.write_text(content, encoding="utf-8")
        return "updated", backup

    path.write_text(content, encoding="utf-8")
    return "created", None


def write_configmap(
    path: Path, name: str, namespace: str, values: dict[str, str]
) -> tuple[str, Path | None]:
    lines = metadata("ConfigMap", f"{name}-configmap", namespace)
    if values:
        lines.append("data:")
        for key in sorted(values):
            lines.append(f"  {key}: {json.dumps(values[key])}")
    else:
        lines.append("data: {}")
    return write_with_backup(path, "\n".join(lines) + "\n")


def write_secret(
    path: Path, name: str, namespace: str, values: dict[str, str]
) -> tuple[str, Path | None]:
    lines = metadata("Secret", f"{name}-secret", namespace)
    lines.append("type: Opaque")
    if values:
        lines.append("data:")
        for key in sorted(values):
            encoded = base64.b64encode(values[key].encode("utf-8")).decode("ascii")
            lines.append(f"  {key}: {encoded}")
    else:
        lines.append("data: {}")
    return write_with_backup(path, "\n".join(lines) + "\n")


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "generated"


def prompt_env_path(initial: Path | None, parser: argparse.ArgumentParser) -> Path:
    prompt = "Path to .env file"
    if initial is not None:
        prompt += f" [{initial}]"
    prompt += ": "

    try:
        entered = input(prompt).strip()
    except EOFError:
        parser.error("a .env path is required")

    if entered:
        path = Path(entered).expanduser()
    elif initial is not None:
        path = initial.expanduser()
    else:
        parser.error("a .env path is required")

    if not path.is_file():
        parser.error(f".env path does not exist or is not a file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a local .env file into Kubernetes ConfigMap and Secret manifests."
    )
    parser.add_argument(
        "--env",
        type=Path,
        help="Optional prompt prefill. The script still asks for the .env path.",
    )
    parser.add_argument("--name", required=True, help="Base Kubernetes resource name")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output directory for generated manifests",
    )
    args = parser.parse_args()

    env_path = prompt_env_path(args.env, parser)
    values, warnings = parse_env(env_path)
    config, secrets = split_values(values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configmap_path = args.output_dir / f"{args.name}-configmap.yaml"
    secret_path = args.output_dir / f"{args.name}-secret.yaml"

    configmap_status, configmap_backup = write_configmap(
        configmap_path, args.name, args.namespace, config
    )
    secret_status, secret_backup = write_secret(
        secret_path, args.name, args.namespace, secrets
    )

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print(f"{configmap_status} {configmap_path}")
    if configmap_backup is not None:
        print(f"backup {configmap_backup}")
    print(f"{secret_status} {secret_path}")
    if secret_backup is not None:
        print(f"backup {secret_backup}")
    print(
        "WARNING: generated Secret manifests contain sensitive values. Do not commit them.",
        file=sys.stderr,
    )
    print(
        "WARNING: review the split before applying; key-name matching is only a heuristic.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
