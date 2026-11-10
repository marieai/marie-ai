from __future__ import annotations

import hashlib
import re
from typing import Any

from marie.auth.api_key_manager import KeyGenerator, str2bool
from marie.excepts import BadConfigSource

# Public examples in APIKeyManager.from_config's documentation.
_EXAMPLE_KEY_DIGESTS = frozenset(
    {
        '2ac8c0da3c9c3c568fc86ef8ca4f934b05044b65d6aae9a4ee2fe0e9bd00feb6',
        'c62607d20f357873f530d453014e1c4ee13da3a85051585ce2ed3a60bc8429cb',
        '8a62dbb45cdf91c20779b86769ed8a186ef0915a7d5b723cf434506eb8c3e016',
        'ea00d927b0ec9663d5c90b51854843b391a8b9181a3213cc30c9cea01620fbc0',
        '7b25e25aa2d2148bfde7de4d81f300d04212c3f8a116339901b46802a869c288',
    }
)
_CONFIG_ERROR = (
    'auth_boot_invalid_configuration: configure auth.keys as a list of named API keys.'
)


def validate_auth_at_startup(auth_config: dict[str, Any] | None) -> None:
    """Reject missing, malformed or published-example gateway credentials.

    Args:
        auth_config: Effective gateway authentication configuration.

    Raises:
        BadConfigSource: Authentication cannot safely be initialized.
    """
    if auth_config is None:
        auth_config = {}
    if not isinstance(auth_config, dict):
        raise BadConfigSource(_CONFIG_ERROR)
    keys = auth_config.get('keys', [])
    if not isinstance(keys, list):
        raise BadConfigSource(_CONFIG_ERROR)

    has_enabled_key = False
    for entry in keys:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get('name'), str)
            or not entry['name'].strip()
        ):
            raise BadConfigSource(_CONFIG_ERROR)
        key = entry.get('api_key')
        if (
            not isinstance(key, str)
            or not KeyGenerator.validate_key(key)
            or re.fullmatch(r'[A-Za-z0-9_-]+', key) is None
        ):
            raise BadConfigSource(
                'auth_boot_invalid_key: supply a generated mas_ or mau_ API key.'
            )
        if hashlib.sha256(key.encode()).hexdigest() in _EXAMPLE_KEY_DIGESTS:
            raise BadConfigSource(
                'auth_boot_example_key: replace published examples with generated API keys.'
            )
        enabled = entry.get('enabled', True)
        if not isinstance(enabled, (bool, str)) or (
            isinstance(enabled, str)
            and enabled.lower()
            not in {'yes', 'y', 'true', 't', '1', 'no', 'n', 'false', 'f', '0'}
        ):
            raise BadConfigSource(_CONFIG_ERROR)
        has_enabled_key = has_enabled_key or str2bool(enabled)

    if not has_enabled_key:
        raise BadConfigSource(
            'auth_boot_no_enabled_keys: configure at least one enabled gateway API key.'
        )
