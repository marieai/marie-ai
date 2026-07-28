from marie.components.document_indexer.llm_task import (
    parse_json_output,
    parse_markdown_json,
    parse_task_output,
)


def test_document_indexer_parser_uses_canonical_repair() -> None:
    value, failed = parse_json_output('```JSON\n{"value": 1,}\n```')

    assert value == {"value": 1}
    assert failed is False


def test_markdown_parser_preserves_failure_tuple_contract() -> None:
    value, failed = parse_markdown_json("no structured output")

    assert value == {"value": "ERROR", "reason": "JSON CONVERSION FAILURE"}
    assert failed is True


def test_parse_task_output_preserves_json_conversion_failure() -> None:
    assert parse_task_output(["no structured output"], "json") == [
        (
            None,
            {"value": "ERROR", "reason": "JSON CONVERSION FAILURE"},
        )
    ]


def test_parse_task_output_preserves_text_without_conversion() -> None:
    assert parse_task_output(["plain text"], "text") == [("plain text", None)]


def test_parse_task_output_accepts_json_arrays() -> None:
    assert parse_task_output(["[1, 2, 3]"], "json") == [([1, 2, 3], None)]
