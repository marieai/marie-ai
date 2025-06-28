import json
from typing import Any

import yaml


def _marshal_llm_to_json(output: str) -> str:
    """
    Extract a substring containing valid JSON or array from a string.

    Args:
        output: A string that may contain a valid JSON object or array surrounded by
        extraneous characters or information.

    Returns:
        A string containing a valid JSON object or array.
    """
    output = output.strip()

    left_square = output.find("[")
    left_brace = output.find("{")

    if left_square < left_brace and left_square != -1:
        left = left_square
        right = output.rfind("]")
    else:
        left = left_brace
        right = output.rfind("}")

    return output[left : right + 1]


def parse_json_markdown(text: str) -> Any:
    if "```json" in text:
        text = text.split("```json")[1].strip().strip("```").strip()

    json_string = _marshal_llm_to_json(text)

    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e_json:
        try:
            import json5

            return json5.loads(text)
        except ValueError:
            try:
                # NOTE: parsing again with pyyaml
                #       pyyaml is less strict, and allows for trailing commas
                #       right now we rely on this since guidance program generates
                #       trailing commas
                json_obj = yaml.safe_load(json_string)
            except yaml.YAMLError as e_yaml:
                raise Exception(
                    f"Got invalid JSON object. Error: {e_json} {e_yaml}. "
                    f"Got JSON string: {json_string}"
                )
            except NameError as exc:
                raise ImportError("Please pip install PyYAML.") from exc

    return json_obj


def parse_markdown_markdown(text: str, return_content=True) -> str:
    """
    Extracts the content enclosed in the first ```markdown ... ``` code block
    from the given text, even if there's a preceding block labeled ```plain text
    or similar.

    If no ```markdown block is found, returns an empty string.
    """
    if not text:
        return "" if return_content else ""

    if "```markdown" not in text:
        return text if return_content else ""
    text = text.split("```markdown")[1].strip().strip("```").strip()
    return text


def check_content_type(text: str) -> str:
    """
    Checks if the given text has code blocks with
    ```json or
    ```markdown.

    Returns:
        "json" if a code block labeled ```json is found.
        "markdown" if ```markdown is found.
        "none" otherwise.
    """
    if not text:
        return "none"

    if "```json" in text:
        return "json"
    elif "```markdown" in text:
        return "markdown"
    else:
        return "none"
