import re
from types import SimpleNamespace

import pytest

from marie.auth.api_key_manager import APIKeyManager, KeyGenerator
from marie.excepts import BadConfigSource


@pytest.fixture(autouse=True)
def empty_key_registry(monkeypatch):
    monkeypatch.setattr(APIKeyManager, '_keys', {})


@pytest.fixture
def auth_setup():
    from marie.utils.server_runtime import setup_auth

    return setup_auth


def credential(*, enabled=True):
    return {
        'name': 'startup-test',
        'api_key': KeyGenerator.generate_key('mas_'),
        'enabled': enabled,
    }


@pytest.mark.parametrize('config', [None, {}, {'keys': []}])
def test_missing_credentials_refuse_startup(auth_setup, config):
    with pytest.raises(BadConfigSource, match='auth_boot_no_enabled_keys'):
        auth_setup(config)
    assert APIKeyManager.get_keys() == {}


@pytest.mark.parametrize('enabled', [False, 'false', '0', 'no'])
def test_disabled_credentials_cannot_satisfy_startup(auth_setup, enabled):
    with pytest.raises(BadConfigSource, match='auth_boot_no_enabled_keys'):
        auth_setup({'keys': [credential(enabled=enabled)]})
    assert APIKeyManager.get_keys() == {}


@pytest.mark.parametrize('prefix', ['mas_', 'mau_'])
def test_generated_credentials_load_with_existing_policy(auth_setup, prefix):
    key = credential()
    key.update(
        api_key=KeyGenerator.generate_key(prefix),
        scopes=['runtime-observability'],
        allowed_fabrics=['fabric-a'],
    )
    auth_setup({'keys': [key]})
    assert APIKeyManager.is_valid(key['api_key'])
    assert APIKeyManager.can_observe_runtime(key['api_key'], 'fabric-a')
    assert not APIKeyManager.can_observe_runtime(key['api_key'], 'fabric-b')


@pytest.mark.parametrize(
    'value', ['', ' ', '${{ ENV.UNSET_KEY }}', None, 123, 'mas_' + '!' * 54]
)
def test_malformed_credential_error_is_redacted(auth_setup, value):
    key = credential()
    key['api_key'] = value
    with pytest.raises(BadConfigSource, match='auth_boot_invalid_key') as caught:
        auth_setup({'keys': [key]})
    if isinstance(value, str) and value.strip():
        assert value not in str(caught.value)
    assert APIKeyManager.get_keys() == {}


def test_published_example_credentials_refuse_startup(auth_setup):
    examples = re.findall(
        r'(?:mas|mau)_[A-Za-z0-9_-]{54}', APIKeyManager.from_config.__doc__
    )
    assert len(examples) == 5
    for example in examples:
        with pytest.raises(BadConfigSource, match='auth_boot_example_key') as caught:
            auth_setup({'keys': [{'name': 'published-example', 'api_key': example}]})
        assert example not in str(caught.value)
    assert APIKeyManager.get_keys() == {}


@pytest.mark.parametrize(
    'config',
    [[], 'invalid', {'keys': {}}, {'keys': [None]}, {'keys': [{'api_key': 'x'}]}],
)
def test_invalid_config_shape_has_a_safe_startup_error(auth_setup, config):
    with pytest.raises(BadConfigSource, match='auth_boot_invalid_configuration'):
        auth_setup(config)


def test_later_invalid_key_never_partially_loads_registry(auth_setup):
    first = credential()
    invalid = credential()
    invalid['api_key'] = 'private-startup-test-marker'
    with pytest.raises(BadConfigSource):
        auth_setup({'keys': [first, invalid]})
    assert not APIKeyManager.is_valid(first['api_key'])


class ListenerSetupReached(Exception):
    pass


def gateway_without_resources(config):
    from marie.serve.runtimes.gateway.marie import MarieServerGateway

    gateway = object.__new__(MarieServerGateway)
    gateway.args = {'auth': config}
    gateway.runtime_args = SimpleNamespace(host='0.0.0.0')
    gateway.logger = SimpleNamespace(debug=lambda *args: None)
    return gateway


@pytest.mark.asyncio
async def test_gateway_rejects_missing_auth_before_listener_setup(monkeypatch):
    from marie.serve.runtimes.servers.composite import CompositeServer

    opened = []

    async def listener_setup(self):
        opened.append(True)
        raise ListenerSetupReached

    monkeypatch.setattr(CompositeServer, 'setup_server', listener_setup)
    gateway = gateway_without_resources({})
    with pytest.raises(BadConfigSource, match='auth_boot_no_enabled_keys'):
        await gateway.setup_server()
    assert opened == []


@pytest.mark.asyncio
async def test_gateway_loads_auth_before_listener_setup(monkeypatch):
    from marie.serve.runtimes.servers.composite import CompositeServer

    key = credential()
    accepted_at_listener_setup = []

    async def listener_setup(self):
        accepted_at_listener_setup.append(APIKeyManager.is_valid(key['api_key']))
        raise ListenerSetupReached

    monkeypatch.setattr(CompositeServer, 'setup_server', listener_setup)
    gateway = gateway_without_resources({'keys': [key]})
    with pytest.raises(ListenerSetupReached):
        await gateway.setup_server()
    assert accepted_at_listener_setup == [True]
