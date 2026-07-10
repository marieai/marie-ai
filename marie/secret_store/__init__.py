"""marie-ai secret resolution — reusable across components.

Resolves `secret_ref` strings (env / mapping / database-backed) to values, plus a
`CredentialResolver` that turns a list of `CredentialRequirement`s into a
``{name: value}`` map. Lives here (not under any one subsystem) so executors,
the scheduler, connectors, and agent tools share one resolver.
"""

from __future__ import annotations

from marie.secret_store.database import DatabaseSecretProvider
from marie.secret_store.providers import (
    EnvSecretProvider,
    MappingSecretProvider,
    SecretProvider,
)
from marie.secret_store.resolver import (
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
