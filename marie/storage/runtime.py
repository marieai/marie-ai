"""Resolve runtime storage references without depending on Studio."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from marie.secret_store.providers import EnvSecretProvider
from marie.secret_store.resolver import CredentialResolver

_PROFILE_QUERY = '''
SELECT c.provider, c.default_bucket, c.region, c.endpoint, c.is_enabled,
       c.access_key_secret_ref, c.secret_key_secret_ref, c.visibility,
       c.organization_id, c.owner_id, c.target_revision
FROM marie_runtime.storage_connections c
WHERE c.id = $1
'''


class StorageReference(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    source: Literal['system', 'connection']
    fabric_group_id: str = Field(min_length=1)
    connection_id: UUID | None = None
    target_revision: int | None = Field(default=None, ge=1)
    target_fingerprint: str | None = None
    organization_id: UUID | None = None
    owner_id: UUID | None = None


@dataclass(frozen=True)
class StorageTarget:
    bucket: str
    region: str
    endpoint_url: str | None
    _reference: dict[str, Any] = field(repr=False)
    access_key: str | None = field(default=None, repr=False)
    secret_key: str | None = field(default=None, repr=False)
    session_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        expected = self._reference.get('target_fingerprint')
        if expected and self.reference['target_fingerprint'] != expected:
            raise ValueError('Storage target fingerprint changed before execution')

    @property
    def reference(self) -> dict[str, Any]:
        descriptor = [self.bucket, self.region, self.endpoint_url or '']
        fingerprint = hashlib.sha256(
            json.dumps(descriptor, separators=(',', ':')).encode()
        ).hexdigest()
        return {**self._reference, 'target_fingerprint': fingerprint}

    def create_client(self) -> Any:
        import boto3
        from botocore.config import Config

        kwargs: dict[str, Any] = {
            'region_name': self.region,
            'config': Config(
                s3={'addressing_style': 'path' if self.endpoint_url else 'auto'}
            ),
        }
        if self.endpoint_url:
            kwargs['endpoint_url'] = self.endpoint_url
        if self.access_key and self.secret_key:
            kwargs['aws_access_key_id'] = self.access_key
            kwargs['aws_secret_access_key'] = self.secret_key
        if self.session_token:
            kwargs['aws_session_token'] = self.session_token
        return boto3.client('s3', **kwargs)


def system_storage_target(reference: dict[str, Any] | None = None) -> StorageTarget:
    bucket = (
        os.getenv('S3_STORAGE_BUCKET_NAME')
        or os.getenv('S3_BUCKET_NAME')
        or os.getenv('S3_BUCKET')
    )
    if not bucket:
        raise ValueError('System storage bucket is not configured')
    access = os.getenv('S3_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID')
    secret = os.getenv('S3_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
    if bool(access) != bool(secret):
        raise ValueError('System storage credentials are incomplete')
    return StorageTarget(
        bucket=bucket,
        region=os.getenv('S3_REGION') or os.getenv('AWS_REGION') or 'us-east-1',
        endpoint_url=os.getenv('S3_ENDPOINT_URL') or None,
        _reference=reference or {'source': 'system'},
        access_key=access,
        secret_key=secret,
        session_token=os.getenv('S3_SESSION_TOKEN') or os.getenv('AWS_SESSION_TOKEN'),
    )


def _parse_reference(
    reference: dict[str, Any], fabric_group_id: str | None
) -> StorageReference:
    try:
        ref = StorageReference.model_validate(reference)
    except ValidationError:
        raise ValueError('Invalid runtime storage reference') from None
    if fabric_group_id is not None and ref.fabric_group_id != fabric_group_id:
        raise ValueError('Storage reference targets another fabric')
    if ref.source == 'connection' and not ref.connection_id:
        raise ValueError('Registered storage requires a connection ID')
    return ref


def _profile_target(
    ref: StorageReference, profile: dict[str, Any] | None
) -> StorageTarget:
    if not profile:
        raise ValueError('Storage connection is unavailable')
    if not profile['is_enabled'] or profile['provider'] != 's3':
        raise ValueError('Storage connection must be an enabled S3 connection')
    visibility = profile['visibility']
    if visibility != 'system':
        if profile['organization_id'] != ref.organization_id:
            raise ValueError('Storage connection access denied')
        if visibility == 'personal' and profile['owner_id'] != ref.owner_id:
            raise ValueError('Storage connection access denied')
    if (
        ref.target_revision is not None
        and ref.target_revision != profile['target_revision']
    ):
        raise ValueError('Storage target revision changed before execution')
    if not profile['default_bucket']:
        raise ValueError('Registered storage bucket is not configured')
    access_ref, secret_ref = (
        profile['access_key_secret_ref'],
        profile['secret_key_secret_ref'],
    )
    if not access_ref or not secret_ref:
        raise ValueError('Registered storage credential references are required')
    if os.getenv('SECRET_PROVIDER', 'env') != 'env':
        raise ValueError(
            'The runtime storage credential provider is not configured for this deployment'
        )
    resolver = CredentialResolver(secret_provider=EnvSecretProvider())
    try:
        access, secret = (
            resolver.resolve_ref(access_ref),
            resolver.resolve_ref(secret_ref),
        )
    except ValueError:
        raise ValueError(
            'Registered storage credential references cannot be resolved'
        ) from None
    if not access or not secret:
        raise ValueError('Registered storage credentials are unavailable')
    reference = ref.model_dump(mode='json', exclude_none=True)
    reference['target_revision'] = profile['target_revision']
    return StorageTarget(
        bucket=profile['default_bucket'],
        region=profile['region'] or 'us-east-1',
        endpoint_url=profile['endpoint'] or None,
        _reference=reference,
        access_key=access,
        secret_key=secret,
    )


def resolve_storage_reference(
    reference: dict[str, Any],
    pool: Any = None,
    *,
    fabric_group_id: str | None = None,
) -> StorageTarget:
    ref = _parse_reference(reference, fabric_group_id)
    if ref.source == 'system':
        target = system_storage_target(ref.model_dump(mode='json', exclude_none=True))
    else:
        if pool is None:
            raise ValueError('Registered storage requires the executor database pool')
        with (
            pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            profile = cursor.execute(
                _PROFILE_QUERY.replace('$1', '%s'), (ref.connection_id,)
            ).fetchone()
        target = _profile_target(ref, profile)
    return target


async def resolve_storage_reference_async(
    reference: dict[str, Any],
    pool: Any = None,
    *,
    fabric_group_id: str | None = None,
) -> StorageTarget:
    ref = _parse_reference(reference, fabric_group_id)
    if ref.source == 'system':
        target = system_storage_target(ref.model_dump(mode='json', exclude_none=True))
    else:
        if pool is None:
            raise ValueError('Registered storage requires the sensor database pool')
        profile = await pool.fetchrow(_PROFILE_QUERY, ref.connection_id)
        target = _profile_target(ref, dict(profile) if profile else None)
    return target
