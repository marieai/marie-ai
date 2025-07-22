import pytest

from marie.extract.annotators.field_extractor import FieldValueExtractor


@pytest.fixture
def labels():
    return [
        "PATIENT NAME",
        "MEMBER NUMBER",
        "CLAIM NUMBER",
        "PROVIDER NPI",
        "CHECK NUMBER",
        "CHECK DATE",
        "TAX ID",
        "PATIENT ACCT"
    ]


@pytest.fixture
def extractor(labels):
    return FieldValueExtractor(target_labels=labels, fuzzy_threshold=0.85)


def test_clean_extraction(extractor):
    line = "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 123456789"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN DOE"


def test_missing_colon(extractor):
    line = "PATIENT NAME JOHN DOE MEMBER NUMBER 123456789"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN DOE"


def test_extra_spaces(extractor):
    line = "PATIENT    NAME :   JOHN    DOE   MEMBER NUMBER : 123456789"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN DOE"


def test_ocr_typo_label(extractor):
    line = "PATIENT NAME: JOHN DOE MEMER NUMBER: 123456789"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN DOE"


def test_only_value_available(extractor):
    line = "PATIENT NAME: JOHN DOE"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN DOE"


def test_no_delimiters_all_glued(extractor):
    line = "PATIENT NAME JOHN DOE PROVIDER NPI 987654321"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN DOE"


def test_short_line(extractor):
    line = "PATIENT NAME: JOHN"
    match_text = "PATIENT NAME"
    value = extractor.extract(line, match_text)
    assert value == "JOHN"


def test_multiple_fields_same_line_001(extractor):
    line = "TAX ID: 12345 CLAIM NUMBER: 98765 CHECK DATE: 01/01/2024"
    match_text = "TAX ID"
    value = extractor.extract(line, match_text)
    assert value == "12345"


def test_multiple_fields_same_line_002(extractor):
    line = "TAX ID: 12345 CLAIM NUMBER: 98765 CHECK DATE: 01/01/2024"
    match_text = "CLAIM NUMBER"
    value = extractor.extract(line, match_text)
    assert value == "98765"


def test_multiple_separators_01(extractor):
    line = "TAX ID: 123456789 PATIENT ACCT #: 600647211111 CLAIM NUMBER: 278022233 PROVIDER NPI: 123456789"
    match_text = "PATIENT ACCT"
    value = extractor.extract(line, match_text)
    assert value == "600647211111"
