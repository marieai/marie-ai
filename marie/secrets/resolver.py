"""marie-ai-native resolution of secret references to values.

Resolution lives ENTIRELY in marie-ai (no calls to Studio or any other system).
A reference carries a `secret_ref` (the same grammar Studio uses), resolved here
against a pluggable `SecretProvider` backend:

    env:NAME  /  env://NAME        -> environment variable NAME
    secrets:NAME                   -> SecretProvider.get_secret(NAME)
    {{$secrets.NAME}}              -> SecretProvider.get_secret(NAME)
    {{$secrets.PROVIDER.NAME}}     -> SecretProvider.get_secret(NAME)  (provider hint ignored)

This lives at `marie.secrets` so any component (executors, scheduler, connectors,
agent tools, …) can reuse it. A real database/Vault backend slots in behind the
same `SecretProvider` protocol later.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from marie.secrets.providers import EnvSecretProvider, SecretProvider

_SECRET_EXPR_RE = re.compile(r"^\{\{\s*\$secrets\.([A-Za-z0-9_.]+)\s*\}\}$")


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class CredentialRequirement(BaseModel):
    """A credential a component needs, plus the bound secret reference (if any).

    Ingests Studio camelCase keys (e.g. `secretRef`) verbatim.
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    name: str
    secret_ref: str | None = None
    required: bool = False


def parse_secret_ref(secret_ref: str) -> tuple[str, str]:
    """Return ``(scheme, name)`` for a secret reference. scheme in {'env','secrets'}."""
    ref = secret_ref.strip()
    if ref.startswith("env://"):
        return "env", ref[len("env://") :]
    if ref.startswith("env:"):
        return "env", ref[len("env:") :]
    if ref.startswith("secrets:"):
        return "secrets", ref[len("secrets:") :]
    match = _SECRET_EXPR_RE.match(ref)
    if match:
        # {{$secrets.NAME}} or {{$secrets.PROVIDER.NAME}} -> take the trailing name.
        return "secrets", match.group(1).split(".")[-1]
    raise ValueError(f"unsupported secret reference scheme: {secret_ref!r}")


class CredentialResolver:
    """Resolves credential requirements to a ``{name: value}`` map, in marie-ai.

    ``env:`` refs resolve from `env_provider` (the environment by default);
    ``secrets:`` / ``{{$secrets.*}}`` refs resolve from `secret_provider`
    (mockable; database/Vault-backed later).
    """

    def __init__(
        self,
        *,
        env_provider: SecretProvider | None = None,
        secret_provider: SecretProvider | None = None,
    ) -> None:
        self._env = env_provider or EnvSecretProvider()
        self._secrets = secret_provider

    def resolve_ref(self, secret_ref: str) -> str | None:
        scheme, name = parse_secret_ref(secret_ref)
        if scheme == "env":
            return self._env.get_secret(name)
        if scheme == "secrets":
            if self._secrets is None:
                raise ValueError(
                    f"no secret provider configured to resolve {secret_ref!r}"
                )
            return self._secrets.get_secret(name)
        raise ValueError(f"unsupported secret scheme {scheme!r}")

    def resolve(self, requirements: list[CredentialRequirement]) -> dict[str, str]:
        out: dict[str, str] = {}
        for req in requirements:
            if not req.secret_ref:
                # Unbound credential: skip (the daemon/plugin treats it as absent).
                if req.required:
                    raise ValueError(
                        f"required credential {req.name!r} has no bound secret_ref"
                    )
                continue
            value = self.resolve_ref(req.secret_ref)
            if value is None:
                if req.required:
                    raise ValueError(
                        f"required credential {req.name!r} could not be resolved "
                        f"from {req.secret_ref!r}"
                    )
                continue
            out[req.name] = value
        return out
