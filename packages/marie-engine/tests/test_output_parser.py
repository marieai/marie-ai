import pytest
from marie.engine import output_parser
from marie.engine.output_parser import (
    JSONOutputParserError,
    check_content_type,
    parse_json_markdown,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"ok": true}', {"ok": True}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('```JSON\n{"ok": true}\n```', {"ok": True}),
        ('```\n{"ok": true}\n```', {"ok": True}),
        ('```json\n{"ok": true', {"ok": True}),
        ('Result: {"ok": true} Done.', {"ok": True}),
        ('<output>{"ok": true}</output>', {"ok": True}),
        ('{"value": 1,}', {"value": 1}),
        ("{'value': 1}", {"value": 1}),
        ("{value: 1}", {"value": 1}),
        ('{"value": "open}', {"value": "open"}),
        ('{"value": 1', {"value": 1}),
        ("[1, 2, 3", [1, 2, 3]),
        ('{"outer": {"inner": 1}', {"outer": {"inner": 1}}),
        ('{"text": "use {x}", "ok": true}', {"text": "use {x}", "ok": True}),
        ('{"a": 1, // comment\n"b": 2}', {"a": 1, "b": 2}),
        ("Result [1, 2, 3] complete.", [1, 2, 3]),
    ],
)
def test_parse_json_markdown_handles_common_model_output_damage(
    text: str, expected: object
) -> None:
    assert parse_json_markdown(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"value": 1,}', {"value": 1}),
        ("{'value': 1}", {"value": 1}),
        ("{value: 1}", {"value": 1}),
        ('{"value": 1, // comment\n}', {"value": 1}),
        ("Thinking complete.\n```json\n{value: 'ready',}\n```", {"value": "ready"}),
    ],
)
def test_json5_syntax_is_parsed_before_repair(
    monkeypatch: pytest.MonkeyPatch, text: str, expected: object
) -> None:
    def fail_repair(*args: object, **kwargs: object) -> object:
        raise AssertionError("json_repair should not parse valid JSON5")

    monkeypatch.setattr(output_parser.json_repair, "loads", fail_repair)

    assert parse_json_markdown(text) == expected


def test_strict_json_is_parsed_before_relaxed_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fallback(*args: object, **kwargs: object) -> object:
        raise AssertionError("fallback parser should not parse strict JSON")

    monkeypatch.setattr(output_parser.json5, "loads", fail_fallback)
    monkeypatch.setattr(output_parser.json_repair, "loads", fail_fallback)

    assert parse_json_markdown('{"value": 1}') == {"value": 1}


def test_expected_root_type_disambiguates_prose_brackets() -> None:
    text = '[analysis] See reference [1]. Final answer: {"ok": true}'

    assert parse_json_markdown(text, expected_type=dict) == {"ok": True}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "No structured output is available.",
        "42",
        '"text"',
        '{"first": 1}{"second": 2}',
        '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```',
        '[analysis] Final answer: {"ok": true}',
    ],
)
def test_parse_json_markdown_rejects_missing_or_ambiguous_output(text: str) -> None:
    with pytest.raises(JSONOutputParserError):
        parse_json_markdown(text)


def test_parse_json_markdown_enforces_expected_root_type() -> None:
    with pytest.raises(JSONOutputParserError, match="root type list"):
        parse_json_markdown("[1, 2, 3]", expected_type=dict)


def test_parse_error_does_not_include_model_output() -> None:
    sensitive_output = "patient-123 has no JSON response"

    with pytest.raises(JSONOutputParserError) as exc_info:
        parse_json_markdown(sensitive_output)

    assert sensitive_output not in str(exc_info.value)


def test_content_type_recognizes_case_insensitive_json_fence() -> None:
    assert check_content_type('```JSON\n{"ok": true}\n```') == "json"
