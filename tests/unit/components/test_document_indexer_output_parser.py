from marie.components.document_indexer.llm_task import (
    PageResult,
    parse_json_output,
    parse_task_output,
)


def test_document_indexer_parser_uses_canonical_repair() -> None:
    result = parse_json_output('```JSON\n{"value": 1,}\n```')

    assert result == PageResult(value={"value": 1})


def test_markdown_parser_preserves_failure_page_result_contract() -> None:
    result = parse_json_output("no structured output")

    assert result == PageResult(
        value={"value": "ERROR", "reason": "JSON CONVERSION FAILURE"},
        error=True,
    )


def test_parse_task_output_preserves_json_conversion_failure() -> None:
    assert parse_task_output(["no structured output"], "json") == [
        PageResult(
            value={"value": "ERROR", "reason": "JSON CONVERSION FAILURE"},
            error=True,
        )
    ]


def test_parse_task_output_preserves_text_without_conversion() -> None:
    assert parse_task_output(["plain text"], "text") == [PageResult(value="plain text")]


def test_parse_task_output_accepts_json_arrays() -> None:
    assert parse_task_output(["[1, 2, 3]"], "json") == [PageResult(value=[1, 2, 3])]
