from typing import Dict, List

import pytest
from sentence_transformers import SentenceTransformer

from marie.extract.annotators.multi_line_matcher import MultiLinePatternMatcher


@pytest.fixture(scope="module")
def model():
    # return SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-en",
        trust_remote_code=True
    )
    # control input sequence length up to 8192
    model.max_seq_length = 1024
    return model


@pytest.fixture

def default_reference_blocks() -> Dict[str, List[str]]:
    """
    Each list element represents a contiguous block of lines to match.
    """
    return {
        "PatternPatient": [
            "TAX ID : 12345ABC PATIENT ACCT : 12345ABC CLAIM NUMBER : 12345ABC PROVIDER NPI : 12345ABC",
            "PATIENT NAME : JOHN SMITH MEMBER NUMBER : 12345ABC CHECK NUMBER : 123456789 CHECK DATE : 04/01/2024",
        ],
        "PatternService": [
            "SERVICE DATE:   04/01/2024 CODE:  12345ABC",
            "DESCRIPTION:  SOME DESCRIPTION TEXT CHARGE:  $0.00",
        ],
        "PatternPayment": [
            "CHECK NUMBER: 12345ABC CHECK DATE:   04/01/2024",
            "PAID AMOUNT:  $0.00",
        ],
    }


@pytest.fixture
def matcher(model, default_reference_blocks, tmp_path):
    import yaml

    multiline_reference_blocks = {
        "multiline_reference_blocks": default_reference_blocks
    }
    config_file = tmp_path / "multiline_reference_blocks.yaml"
    yaml_content = yaml.dump(multiline_reference_blocks, default_flow_style=False)
    config_file.write_text(yaml_content)

    print('config_file path:', config_file)

    return MultiLinePatternMatcher(model, threshold=0.8, reference_blocks=default_reference_blocks)


def test_no_match_when_dissimilar(matcher):
    lines = [
        "COMPLETELY RANDOM TEXT THAT DOES NOT MATCH",
        "ANOTHER RANDOM LINE WITH NO STRUCTURE"
    ]

    matches = matcher.find_matching_blocks(lines, window=2)
    assert len(matches) == 0


def test_full_match(matcher):
    lines = [
        "TAX ID: 123456789    PATIENT ACCT: XYZ    CLAIM NUMBER: 987654321    PROVIDER NPI: 54321",
        "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024",
    ]
    matches = matcher.find_matching_blocks(lines, window=2)
    print(matches)
    assert len(matches) == 1
    assert matches[0]['pattern'] == "PatternPatient"

def test_match_service_block(matcher):
    lines = [
        "SERVICE DATE: 01/01/2023 CODE: 12345",
        "DESCRIPTION: Consultation CHARGE: $150.00"
    ]

    matches = matcher.find_matching_blocks(lines, window=2)

    assert len(matches) > 0
    assert matches[0]["pattern"] == "PatternService"
    assert matches[0]["score"] >= 0.8
    assert matches[0]["start_line"] == 1
    assert matches[0]["end_line"] == 2


def test_match_payment_block(matcher):
    lines = [
        "CHECK NUMBER: 987654 CHECK DATE: 02/02/2024",
        "PAID AMOUNT: $1234.56"
    ]

    matches = matcher.find_matching_blocks(lines, window=2)

    assert len(matches) > 0
    assert matches[0]["pattern"] == "PatternPayment"
    assert matches[0]["score"] >= 0.8
    assert matches[0]["start_line"] == 1
    assert matches[0]["end_line"] == 2


def test_partial_match_should_fail(matcher):
    lines = [
        "TAX ID: 123456789    PATIENT ACCT: ABC123",
        "PATIENT NAME: JOHN DOE    CHECK NUMBER: 888777666"
    ]
    matches = matcher.find_matching_blocks(lines, window=2)
    assert matches == []


def test_no_match_random_lines(matcher):
    lines = [
        "Some completely unrelated text",
        "Another block of noise",
        "Yet more garbage unrelated to structured patterns"
    ]
    matches = matcher.find_matching_blocks(lines)
    assert matches == []


def test_dynamic_addition_of_reference_block(model):
    matcher = MultiLinePatternMatcher(model=model, threshold=0.8)
    matcher.add_reference_block("CustomPattern", (
        "INVOICE DATE : 04/06/2024 INVOICE AMOUNT : $1234.56"
    ))

    lines = [
        "INVOICE DATE: 03/15/2023 INVOICE AMOUNT: $789.00"
    ]

    matches = matcher.find_matching_blocks(lines, window=1)

    assert len(matches) > 0
    assert matches[0]["pattern"] == "CustomPattern"
    assert matches[0]["score"] >= 0.8
    assert matches[0]["start_line"] == 1
    assert matches[0]["end_line"] == 1


def test_mixed_lines_one_match(matcher):
    lines = [
        "RANDOM INTRO",
        "TAX ID: 999999999    PATIENT ACCT: ZZ111    CLAIM NUMBER: 222333444    PROVIDER NPI: 12345",
        "PATIENT NAME: JANE DOE    MEMBER NUMBER: X001    CHECK NUMBER: 555666777    CHECK DATE: 12/12/2023",
        "UNRELATED LINE"
    ]
    matches = matcher.find_matching_blocks(lines, window=2)
    print(matches)
    assert any(m['pattern'] == "PatternPatient" for m in matches)


def test_multi_line_pattern_match(matcher):
    lines = [
        "TAX ID: 123456789    PATIENT ACCT: ABC123    CLAIM NUMBER: 987654321    PROVIDER NPI: 54321",
        "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024",
        "RANDOM LINE NOT MATCHING"
    ]

    matches = matcher.find_matching_blocks(lines, window=2)

    print(matches)

    assert len(matches) == 1
    assert matches[0]['pattern'] == "PatternPatient"
    assert matches[0]['start_line'] == 1
    assert matches[0]['end_line'] == 2


def test_multi_line_pattern_match_multiple(matcher):
    lines = [
        "TAX ID: 123456789    PATIENT ACCT: ABC123    CLAIM NUMBER: 987654321    PROVIDER NPI: 54321",
        "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024",
        "RANDOM LINE NOT MATCHING",
        "ANOTHER RANDOM LINE",
        "YET ANOTHER RANDOM LINE",
        "TAX ID: 45678910    PATIENT ACCT: ABC123    NUMBER: 987654321    PROVIDER NPI: 54321 GROUP: 12345",
        "PATIENT NAME: GREG DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024",
    ]

    matches = matcher.find_matching_blocks(lines, window=2)
    print(matches)

    assert len(matches) == 2
    assert matches[0]['pattern'] == "PatternPatient"
    assert matches[0]['start_line'] == 1
    assert matches[0]['end_line'] == 2

    assert matches[1]['pattern'] == "PatternPatient"
    assert matches[1]['start_line'] == 6
    assert matches[1]['end_line'] == 7


@pytest.mark.parametrize("name,lines,expected", [
    ("perfect_match_block", [
        "TAX ID: 123456789    PATIENT ACCT: ABC123    CLAIM NUMBER: 987654321    PROVIDER NPI: 54321",
        "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024"
    ], 1),
    ("partial_field_order_change", [
        "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024",
        "TAX ID: 123456789    PATIENT ACCT: ABC123    CLAIM NUMBER: 987654321    PROVIDER NPI: 54321"
    ], 1),
    ("extra_unrelated_lines", [
        "RANDOM NOISE LINE THAT SHOULD NOT MATCH",
        "TAX ID: 999999999    PATIENT ACCT: ZZZ111    CLAIM NUMBER: 123123123    PROVIDER NPI: 98765",
        "PATIENT NAME: ALICE SMITH    MEMBER NUMBER: 111222333    CHECK NUMBER: 999888777    CHECK DATE: 02/02/2024",
        "ANOTHER NOISE LINE"
    ], 1),
    ("no_matching_lines", [
        "HELLO WORLD",
        "THIS IS A TEST LINE",
        "IT SHOULD NOT MATCH ANYTHING"
    ], 0),
])
def test_multi_line_patterns(matcher, name, lines, expected):
    matches = matcher.find_matching_blocks(lines, window=2)
    assert len(matches) == expected
