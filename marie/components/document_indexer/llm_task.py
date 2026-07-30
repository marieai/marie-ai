import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.engine.output_parser import JSONOutputParserError, parse_json_markdown
from marie.logging_core.predefined import default_logger as logger
from marie.utils.json import load_json_file


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
    max_tokens: Optional[int] = 4096 * 2
    multimodal: bool = True
    tasks: List[LLMTask]
    debug: Optional[Dict] = None


class PageResult(BaseModel):
    value: Any
    page: int = 0
    error: bool = False


def md_wrap(preformatted_text, format_type: str = None):
    ft = "" if format_type is None else format_type
    return f"```{ft}\n{preformatted_text}\n```"

def _append_ocr_lines(
    prompt: str, words: List[str] = None, lines: List[int] = None, **kwargs
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
    for word, line_num in zip(words, lines):
        line_map[line_num].append(word)

    numbered_lines = [
        f"{line} | {' '.join(line_map[line])}" for line in sorted(line_map)
    ]
    return f"{prompt}\n{md_wrap('\n'.join(numbered_lines))}"

def parse_json_output(json_output: str) -> PageResult:
    try:
        return PageResult(value=parse_json_markdown(json_output))
    except JSONOutputParserError:
        return PageResult(value={"value": "ERROR", "reason": "JSON CONVERSION FAILURE"}, error=True)

PROMPT_STRATEGIES = {
    "prompt_identity": lambda prompt, *args, **kwargs: prompt,
    "append_chained_output": lambda prompt, text="", **kwargs: f"{prompt}\n{text}",
    "append_ocr_lines": _append_ocr_lines,
}

CONVERSION_STRATEGIES = {
    "text": lambda text: PageResult(value=text),
    "json": parse_json_output,
}

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


def parse_task_output(task_outputs: List[str], output_type) -> List[PageResult]:
    return [CONVERSION_STRATEGIES[output_type](task_output) for task_output in task_outputs]


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
    task_outputs: List[PageResult],
    task_output_mod: Optional[LLMOutputModifier] = None
) -> List[PageResult]:
    if task_output_mod is None:
        return task_outputs

    pattern, substitute = task_output_mod.pattern, task_output_mod.substitute
    for task_output in task_outputs:
        task_output.value = modify_task(task_output.value, pattern, substitute)
    return task_outputs


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
    # NOTE: Add other filtering methods as needed
    # elif isinstance(pattern, str):
    #     pass

    return sorted(page_output.keys())
