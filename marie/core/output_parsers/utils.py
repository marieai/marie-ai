import re
from typing import Any, List

from marie.engine import output_parser as engine_output_parser

from marie.core.output_parsers.base import OutputParserException


def parse_json_markdown(
    text: str,
    *,
    expected_type: type | tuple[type, ...] = (dict, list),
) -> Any:
    try:
        return engine_output_parser.parse_json_markdown(
            text, expected_type=expected_type
        )
    except engine_output_parser.JSONOutputParserError as exc:
        raise OutputParserException(str(exc)) from exc


def parse_code_markdown(text: str, only_last: bool) -> List[str]:
    # Regular expression pattern to match code within triple-backticks
    pattern = r"```(.*?)```"

    # Regular expression pattern to match code within triple backticks with
    # a Python marker. Like: ```python df.columns```
    python_str_pattern = re.compile(r"^```python", re.IGNORECASE)
    text = python_str_pattern.sub("```", text)

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the last matched group if requested
    code = matches[-1] if matches and only_last else matches

    # If empty we optimistically assume the output is the code
    if not code:
        # we want to handle cases where the code may start or end with triple
        # backticks
        # we also want to handle cases where the code is surrounded by regular
        # quotes
        # we can't just remove all backticks due to JS template strings

        candidate = text.strip()

        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]

        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1]

        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]

        # For triple backticks we split the handling of the start and end
        # partly because there can be cases where only one and not the other
        # is present, and partly because we don't need to be so worried
        # about it being a string in a programming language
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*", "", candidate)

        if candidate.endswith("```"):
            candidate = candidate[:-3]
        code = [candidate.strip()]

    return code


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")

    return match.group()
