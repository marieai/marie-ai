from __future__ import annotations

import pytest

from marie.secret_store import (
    CredentialRequirement,
    CredentialResolver,
    DatabaseSecretProvider,
    EnvSecretProvider,
    MappingSecretProvider,
    SecretProvider,
    parse_secret_ref,
)


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("env:JINA_API_KEY", ("env", "JINA_API_KEY")),
        ("env://JINA_API_KEY", ("env", "JINA_API_KEY")),
        ("secrets:jina_key", ("secrets", "jina_key")),
        ("{{$secrets.jina_key}}", ("secrets", "jina_key")),
        ("{{$secrets.vault.jina_key}}", ("secrets", "jina_key")),
    ],
)
def test_parse_secret_ref(ref, expected):
    assert parse_secret_ref(ref) == expected


def test_parse_secret_ref_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        parse_secret_ref("plain-value")


def test_requirement_ingests_studio_camelcase():
    req = CredentialRequirement.model_validate(
        {"name": "api_key", "secretRef": "env:JINA_API_KEY", "required": True}
    )
    assert req.name == "api_key"
    assert req.secret_ref == "env:JINA_API_KEY"
    assert req.required is True


def test_resolve_env_ref():
    resolver = CredentialResolver(
        env_provider=EnvSecretProvider({"JINA_API_KEY": "jk-123"})
    )
    creds = resolver.resolve(
        [CredentialRequirement(name="api_key", secret_ref="env:JINA_API_KEY")]
    )
    assert creds == {"api_key": "jk-123"}


def test_resolve_secrets_ref_via_mapping_backend():
    resolver = CredentialResolver(
        env_provider=EnvSecretProvider({}),
        secret_provider=MappingSecretProvider({"jina_key": "from-store"}),
    )
    creds = resolver.resolve(
        [CredentialRequirement(name="api_key", secret_ref="{{$secrets.jina_key}}")]
    )
    assert creds == {"api_key": "from-store"}


def test_optional_unresolved_is_skipped():
    resolver = CredentialResolver(env_provider=EnvSecretProvider({}))
    # optional + missing env -> skipped (not an error)
    creds = resolver.resolve(
        [CredentialRequirement(name="api_key", secret_ref="env:MISSING", required=False)]
    )
    assert creds == {}


def test_required_unresolved_raises():
    resolver = CredentialResolver(env_provider=EnvSecretProvider({}))
    with pytest.raises(ValueError, match="required credential"):
        resolver.resolve(
            [CredentialRequirement(name="api_key", secret_ref="env:MISSING", required=True)]
        )


def test_unbound_optional_is_skipped():
    resolver = CredentialResolver()
    creds = resolver.resolve([CredentialRequirement(name="api_key", secret_ref=None)])
    assert creds == {}


def test_empty_requirements_yield_no_credentials():
    resolver = CredentialResolver()
    assert resolver.resolve([]) == {}


def test_database_provider_is_a_stub():
    # Conforms to the protocol but is deliberately unimplemented until the
    # marie-studio secrets are ported to a neutral (non-studio) schema.
    provider = DatabaseSecretProvider()
    assert isinstance(provider, SecretProvider)
    with pytest.raises(NotImplementedError, match="ported"):
        provider.get_secret("anything")
