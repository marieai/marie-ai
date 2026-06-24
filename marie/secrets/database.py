"""Stub for a future database-backed secret provider.

marie-ai MUST NOT depend on, or be aware of, marie-studio. Secret VALUES (and the
credential bindings that reference them) currently live in marie-studio's schema.
Before marie-ai can resolve secrets from a database, those secrets must be PORTED
to a neutral, non-marie-studio schema that marie-ai owns or shares — there is no
`MARIE_STUDIO_DB_*` coupling here by design.

Until that port lands this is a deliberate stub: it conforms to the
`SecretProvider` protocol but resolves nothing. Use `EnvSecretProvider` (env) or
`MappingSecretProvider` (in-memory / test) today.
"""

from __future__ import annotations


class DatabaseSecretProvider:
    """Placeholder for the future neutral-schema secret store. Resolves nothing yet."""

    def get_secret(self, key: str) -> str | None:
        raise NotImplementedError(
            "DatabaseSecretProvider is a stub: marie-ai does not read secrets from a "
            "database yet. The marie-studio secrets must first be ported to a neutral "
            "(non-marie-studio) schema — marie-ai must not depend on marie-studio. "
            "Use EnvSecretProvider or MappingSecretProvider for now."
        )
