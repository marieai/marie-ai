import pytest

from marie.extract.engine.transform import convert_to_decimal_money


@pytest.mark.parametrize(
    "field_value, expected_result",
    [
        (None, 0.0),  # Null case
        ("", 0.0),  # Empty string
        ("$100.45", 100.45),  # Currency format
        ("-500", -500.0),  # Negative number with "-"
        ("(500)", -500.0),  # Negative number with "()"
        ("1,000.45", 1000.45),  # Comma-separated number
        ("200", 200.0),  # Simple integer
        ("20 0", 20.0),  # Space between digits (assumed typo)
        ("0.20", 0.20),  # Decimal with leading zero
        ("$20.200", 20.20),  # Decimal with 3 fractional digits
        ("-1,000.65", -1000.65),  # Negative number with comma
    ],
)
def test_convert_to_decimal_money(field_value, expected_result):
    assert convert_to_decimal_money(field_value) == pytest.approx(expected_result, 0.01)


@pytest.mark.parametrize(
    "field_value, expected_result",
    [
        ("100.4533", 100.45),
        ("200.5466", 200.55),
    ],
)
def test_convert_to_decimal_money_rounding(field_value, expected_result):
    assert convert_to_decimal_money(field_value) == pytest.approx(expected_result, 0.01)


@pytest.mark.parametrize(
    "field_value",
    [
        ("Not a number0.20"),
        ("A bunch of text"),
    ],
)
def test_convert_to_decimal_money_non_numeric(field_value):
    assert convert_to_decimal_money(field_value) == 0.0
