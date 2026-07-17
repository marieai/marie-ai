from marie.extension import NetworkPermissions, SecretPermissions


def test_network_denies_by_default() -> None:
    permissions = NetworkPermissions()

    assert permissions.is_host_allowed("example.com") is False


def test_network_enabled_empty_allowlist_denies() -> None:
    permissions = NetworkPermissions(enabled=True)

    assert permissions.is_host_allowed("example.com") is False


def test_network_wildcard_requires_trusted_level() -> None:
    permissions = NetworkPermissions(enabled=True, allowedHosts=["*"])

    assert permissions.is_host_allowed("example.com", "community") is False
    assert permissions.is_host_allowed("example.com", "builtin") is True


def test_network_allows_exact_and_subdomain_hosts() -> None:
    permissions = NetworkPermissions(enabled=True, allowedHosts=["example.com"])

    assert permissions.is_host_allowed("example.com") is True
    assert permissions.is_host_allowed("api.example.com") is True
    assert permissions.is_host_allowed("example.net") is False


def test_secret_enabled_empty_allowlist_denies() -> None:
    permissions = SecretPermissions(enabled=True)

    assert permissions.is_secret_allowed("API_KEY") is False


def test_secret_wildcard_requires_trusted_level() -> None:
    permissions = SecretPermissions(enabled=True, allowed=["*"])

    assert permissions.is_secret_allowed("API_KEY", "community") is False
    assert permissions.is_secret_allowed("API_KEY", "system") is True
