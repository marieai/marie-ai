#!/usr/bin/env python3
"""Create credentials for the Compose Valkey service without printing secrets."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import os
import re
import secrets
from pathlib import Path


def _connection_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
        return f'[{address}]' if address.version == 6 else str(address)
    except ValueError:
        if len(host) > 253 or any(
            not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
            for label in host.split('.')
        ):
            raise argparse.ArgumentTypeError(f'invalid Valkey host: {host!r}') from None
        return host


def main() -> None:
    """Generate one full-access account and its client connection file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--host', type=_connection_host, default='localhost')
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    os.umask(0o077)
    try:
        directory.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        parser.error(f'output directory already exists: {directory}')
    (directory / '.gitignore').write_text('*\n')
    server, clients = directory / 'server', directory / 'clients'
    server.mkdir(mode=0o700)
    clients.mkdir(mode=0o700)
    password = secrets.token_urlsafe(48)
    (server / 'password').write_text(password)
    digest = hashlib.sha256(password.encode()).hexdigest()
    (server / 'users.acl').write_text(
        f'user default off\nuser marie on #{digest} ~* &* +@all\n'
    )
    (clients / 'queue-url').write_text(f'redis://marie:{password}@{args.host}:6379/0\n')
    print(f'Created Valkey authentication material: {directory}')
    print(f'Client connection file: {clients / "queue-url"}')


if __name__ == '__main__':
    main()
