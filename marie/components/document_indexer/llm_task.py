import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from marie.logging_core.predefined import default_logger as logger

from ...utils.json import deserialize_value, load_json_file, to_json


def identity(x, *args, **kwargs) -> str:
    return x


def append_ocr_lines(
    prompt: str, words: List[str] = None, boxes=None, lines: List[int] = None, **kwargs
) -> str:
    """
    Appends line-numbered OCR text to a base prompt.

    Args:
        prompt (str): The original prompt.
        words (List[str]): Words to organize into lines.
        lines (List[int]): Corresponding line numbers for each word.

    Returns:
        str: The combined string with numbered lines.
    """

    if not words or not lines:
        return prompt

    line_map = defaultdict(list)
    for word, line_num, box in zip(words, lines, boxes):
        line_map[line_num].append((word, box))

    # Spatially algin words into lines
    char_width = 8.44
    width = max(x + w for (x, y, w, h) in boxes)
    cols = int(width // char_width)
    x_space = np.arange(0, width, 1)
    bins = np.linspace(0, width, cols)
    bins = np.array(bins).astype(np.int32)
    x_hist = np.digitize(x_space, bins, right=True)
    max_characters_per_line = cols

    buffer = []
    min_indent = max_characters_per_line
    for line_num, line_words in line_map.items():
        line_buffer = " " * max_characters_per_line
        for idx, word in enumerate(line_words):
            text, curr_box = line_words[idx]
            x2, y2, w2, h2 = curr_box
            grid_space = x_hist[x2]
            line_buffer = line_buffer[:grid_space] + text + line_buffer[grid_space:]

        buffer.append((line_num, line_buffer.rstrip()))
        indent = len(line_buffer) - len(line_buffer.lstrip())
        if indent < min_indent:
            min_indent = indent

    # Remove max leading whitespace for all lines.
    if min_indent > 0:
        buffer = [(n, line[min_indent:]) for n, line in buffer]
    buffer = "\n".join([f"{n} | {line}" for n, line in buffer])

    return f"{prompt}\n{md_wrap(buffer)}"


def text_append(prompt, append_text="", **kwargs) -> str:
    if not append_text:
        return prompt
    return f"{prompt}\n{append_text}"


def md_wrap(preformatted_text, format_type: str = None):
    ft = "" if format_type is None else format_type
    return f"```{ft}\n{preformatted_text}\n```"


def parse_markdown_json(markdown_json_str: str):
    """
    Extract and parse JSON from a Markdown code block.

    Args:
        markdown_json_str (str): A string that may contain a JSON code block,
                                 typically wrapped in triple backticks with a "json" language tag.

    Returns:
        tuple: (parsed_data, success_flag)
            - parsed_data (dict/list): Parsed JSON data if successful, None otherwise.
            - fail_flag (bool): True if JSON parsing failed, False otherwise.
    """
    # Extract the JSON content between the ```json and ``` markers using a regex.
    match = re.search(r'```json\s*(.*?)\s*```', markdown_json_str, re.DOTALL)
    if match:
        json_content = match.group(1)
    else:
        logger.debug("JSON not surrounded by Markdown formatting")
        json_content = markdown_json_str

    try:
        return json.loads(json_content), False
    except json.JSONDecodeError:
        logger.warning(f"Cannot Convert to JSON:\n{json_content}")
        return {"value": "ERROR", "reason": "JSON CONVERSION FAILURE"}, True


def parse_json_output(json_output: str):
    try:
        return json.loads(json_output), False
    except json.JSONDecodeError:
        return parse_markdown_json(json_output)


PROMPT_STRATEGIES = {
    "prompt_identity": identity,
    "append_ocr_lines": append_ocr_lines,
    "append_chained_output": text_append,
}

CONVERSION_STRATEGIES = {
    "text": identity,
    "json": parse_json_output,
}


class LLMOutputModifier(BaseModel):
    pattern: str
    substitute: str | None


class PageFilter(BaseModel):
    task: str
    pattern: dict | str


class LLMTask(BaseModel):
    name: str
    prompt: str
    prompt_mod_strategy: Optional[str] = "prompt_identity"
    guided_json_schema: Optional[str] = None
    chained_tasks: Optional[List[str]] = []
    output_type: Optional[str] = "text"
    store_results: Optional[bool] = True
    output_mod: Optional[LLMOutputModifier] = None
    page_filter: Optional[PageFilter] = None


class LLMConfig(BaseModel):
    name_or_path: str = Field(alias="_name_or_path")
    engine_provider: Optional[str] = "vllm"
    tasks: List[LLMTask]
    debug: Optional[Dict] = None


def initialize_tasks(tasks: List[LLMTask], task_files_path: str):
    # TODO: Ensure tasks can execute in current or ensure it with an execution graph
    for task in tasks:
        logger.info(f"Initializing LLM task {task.name}")
        # Validate Task Chaining
        if task.chained_tasks:
            if task.name in task.chained_tasks:
                raise ValueError(f"Infinite Task loop for {task.name}")
            for task_name in task.chained_tasks:
                if not any(task_name == task.name for task in tasks):
                    raise ValueError(
                        f"Chained task `{task_name}` not found in configuration"
                    )

        # Validate and load Prompts
        prompt_path = os.path.join(task_files_path, task.prompt)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Unable to locate prompt {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            task.prompt = prompt_file.read()

        # Validate and load JSON schema
        if task.guided_json_schema:
            schema_path = os.path.join(task_files_path, task.guided_json_schema)
            if not os.path.exists(schema_path):
                raise FileNotFoundError(f"Unable to locate schema {schema_path}")
            task.guided_json_schema = load_json_file(schema_path)
            task.output_type = "json"

        # Validate Output types
        if task.output_type not in CONVERSION_STRATEGIES:
            raise NotImplementedError(f"Unknown output type {task.output_type}")


def parse_task_output(task_outputs: List[str], output_type):
    extract_predictions = []
    for task_output in task_outputs:
        prediction, conversion_failure = CONVERSION_STRATEGIES[output_type](task_output)
        error_data = None
        if conversion_failure:
            prediction, error_data = None, prediction

        extract_predictions.append((prediction, error_data))

    return extract_predictions


def modify_task(task, pattern, substitute):
    if isinstance(task, dict):
        return modify_task_dict(task, pattern, substitute)
    elif isinstance(task, list):
        return modify_task_list(task, pattern, substitute)
    elif isinstance(task, str):
        match = re.search(pattern, task, re.DOTALL)
        if match:
            if substitute is None:
                return None
            return re.sub(pattern, substitute, task, re.DOTALL)
        return task
    logger.warning(f"Unable to modify {task} with type {type(task)}")
    return task


def modify_task_list(task_list: List, pattern, substitute):
    return [modify_task(task, pattern, substitute) for task in task_list]


def modify_task_dict(task_dict: Dict, pattern, substitute):
    return {
        key: modify_task(value, pattern, substitute) for key, value in task_dict.items()
    }


def modify_outputs(
    task_outputs: List[str | Dict | List], task_output_mod: LLMOutputModifier = None
):
    if task_output_mod is None:
        return task_outputs
    pattern, substitute = task_output_mod.pattern, task_output_mod.substitute
    return [
        (modify_task(task_output, pattern, substitute), error_data)
        for task_output, error_data in task_outputs
    ]


def filter_pages(pattern: Any, page_output: dict[int, Any], page_subset=None):
    """
    Filters pages that match a given pattern from a provided page output dictionary.

    This function processes a given pattern and filters the pages from a page output dictionary.
    It checks the presence of the pattern in the content of the page's dictionary output and
    returns a sorted list of matching page numbers. A subset of pages to filter can optionally
    be provided.

    Arguments:
        pattern: Any
            The pattern to match against the page output. Currently, must be a dictionary.
        page_output: dict[int, Any]
            A dictionary where keys are page numbers and values are tuples of page-specific
            data and an error indicator.
        page_subset: Optional[Iterable[int]]
            An optional subset of page numbers to filter from. If None, all pages are considered.

    Returns:
        list[int]: A sorted list of page numbers that match the given pattern.
    """
    if isinstance(pattern, dict):
        matching_pages = set()
        pages = page_subset or page_output.keys()
        for page_num, (page_output, error) in page_output.items():
            if page_num not in pages:
                continue
            if not isinstance(page_output, dict):
                continue
            if all(page_output.get(k) == v for k, v in pattern.items()):
                matching_pages.add(page_num)
        return sorted(matching_pages)
    # TODO: if needed
    # elif isinstance(pattern, str):
    #     pass

    return sorted(page_output.keys())
