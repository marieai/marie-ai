"""marie-ai secret resolution — reusable across components.

Resolves `secret_ref` strings (env / mapping / database-backed) to values, plus a
`CredentialResolver` that turns a list of `CredentialRequirement`s into a
``{name: value}`` map. Lives here (not under any one subsystem) so executors,
the scheduler, connectors, and agent tools share one resolver.
"""

from __future__ import annotations

from marie.secrets.database import DatabaseSecretProvider
from marie.secrets.providers import (
    EnvSecretProvider,
    MappingSecretProvider,
    SecretProvider,
)
from marie.secrets.resolver import (
    CredentialRequirement,
    CredentialResolver,
    parse_secret_ref,
)

__all__ = [
    "SecretProvider",
    "EnvSecretProvider",
    "MappingSecretProvider",
    "DatabaseSecretProvider",
    "CredentialRequirement",
    "CredentialResolver",
    "parse_secret_ref",
]
