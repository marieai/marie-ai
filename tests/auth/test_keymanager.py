import pytest


def test_key_generation():
    """Test key generation"""

    from marie_server.auth.api_key_manager import KeyGenerator
    key = KeyGenerator.generate_key(prefix="mas_")

    print(key)
    assert key.startswith("mas_")
    assert KeyGenerator.validate_key(key)
