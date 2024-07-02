import json

import pytest

from marie.serve.discovery.address import JsonAddress, PlainAddress


@pytest.mark.parametrize(
    'addr, exp_addr', (
            ('1.2.3.4', '1.2.3.4'),
            ('5.6.7.8', '5.6.7.8'),
    )
)
def test_to_plain_address(addr, exp_addr):
    assert PlainAddress(addr).add_value() == exp_addr
    assert PlainAddress(addr).delete_value() == exp_addr


@pytest.mark.parametrize(
    'addr, exp_addr', (
            (b'1.2.3.4', '1.2.3.4'),
            (b'5.6.7.8', '5.6.7.8'),
            (b'11.2.3.4', '11.2.3.4'),
            (b'55.6.7.8', '55.6.7.8'),
    )
)
def test_from_plain_address(addr, exp_addr):
    assert PlainAddress.from_value(addr) == exp_addr


@pytest.mark.parametrize(
    'val', (
            '1.2.3.4',
            '5.6.7.8',
    )
)
def test_to_json_address(val):
    assert JsonAddress(val).add_value() == json.dumps({
        'Op': 0, 'Addr': val, 'Metadata': "{}"})
    assert JsonAddress(
        val, metadata={'name': 'host1'}).delete_value() == json.dumps({
        'Op': 1, 'Addr': val, 'Metadata': json.dumps({'name': 'host1'})})


@pytest.mark.parametrize(
    'val, op, addr', (
            (b'{"Op": 1, "Addr": "1.2.3.4", "Metadata": "{}"}', False, '1.2.3.4'),
            (b'{"Op": 0, "Addr": "5.6.7.8", "Metadata": "{}"}', True, '5.6.7.8'),
            (b'{"Op": 1, "Addr": "11.2.3.4", "Metadata": "{}"}', False, '11.2.3.4'),
            (b'{"Op": 0, "Addr": "55.6.7.8", "Metadata": "{}"}', True, '55.6.7.8'),
    )
)
def test_from_json_address(val, op, addr):
    data = JsonAddress.from_value(val)
    assert (op, addr) == data
