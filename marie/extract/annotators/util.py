import asyncio
import json
import logging
import os
import os.path
import threading
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
)

from numpy import ndarray
from PIL import Image
from qwen_vl_utils import smart_resize

from marie.components.document_taxonomy.verbalizers import verbalizers
from marie.engine import EngineLM
from marie.engine.llm_ops import LLMCall
from marie.engine.multimodal_ops import MultimodalLLMCall
from marie.engine.openai_engine import OpenAIEngine
from marie.engine.output_parser import (
    check_content_type,
    parse_json_markdown,
    parse_markdown_markdown,
)
from marie.extract.structures.unstructured_document import UnstructuredDocument
from marie.helper import run_async
from marie.logging_core.predefined import default_logger as logger
from marie.utils.docs import frames_from_file
from marie.utils.utils import batchify

if TYPE_CHECKING:
    from marie.extract.annotators.context_provider import (
        ContextProviderManager,
        ProcessingUnit,
    )

SYSTEM_PROMPT = ""  # Placeholder for the system prompt


def _extract_page_number_from_filename(filename: str) -> Optional[int]:
    """
    Extract page number from filename.

    Expected patterns:
    - frame_0001.png → 1
    - frame_0001.tif → 1
    - page_1.png → 1

    Args:
        filename: The filename to parse

    Returns:
        Page number (1-indexed) or None if not found
    """
    import re

    # Remove known extensions
    base = filename
    for ext in [".png", ".tif", ".json", ".md", ".txt"]:
        if base.lower().endswith(ext):
            base = base[: -len(ext)]

    # Try to extract number from the end
    # Match patterns like: frame_0001, page_1, 0001, etc.
    match = re.search(r"_?(\d+)$", base)
    if match:
        return int(match.group(1))

    return None


def load_prompt(prompt_file: str) -> str:
    """Load the prompt text from a file.
    :param prompt_file: Path to the prompt file.
    :return: The prompt text as a string.
    """
    try:
        with open(os.path.expanduser(prompt_file), "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        return prompt
    except FileNotFoundError:
        logger.error(f"Unable to find the file: {prompt_file}")
        raise


def route_llm_engine(model_name: str, is_multimodal: bool) -> EngineLM:
    """
    Route the LLM call to the appropriate engine based on the model name.
    :param model_name: The name of the model to use.
    :param is_multimodal: Flag indicating if the model is multimodal.
    :return: An instance of the appropriate EngineLM class.
    """

    # this should be set in the environment variables or in .env file
    api_key = os.environ['OPENAI_API_KEY']
    api_base = os.environ['OPENAI_API_BASE']

    logger.info(f'OPENAI_API_KEY = {api_key}')
    logger.info(f'OPENAI_API_BASE = {api_base}')

    if not api_key or not api_base:
        raise ValueError(
            "Both OPENAI_API_KEY and OPENAI_API_BASE must be set in the environment variables."
        )

    engine = OpenAIEngine(
        api_key=api_key,
        base_url=api_base,
        model_name=model_name,
        is_multimodal=is_multimodal,
        cache=False,
    )

    return engine


def preprocess_images_for_inference(
    image_paths, size_factor=28, min_pixels=4 * 28 * 28, max_pixels=1280 * 28 * 28
) -> list:
    """
    Preprocesses a list of images for inference.

    :param image_paths: A list of file paths to the images.
    :param output_dir: Directory path to save the preprocessed images.
    :param prompt: Any prompt or metadata string related to the inference process.
    :param engine: The inference engine to be used
    :param size_factor: Resize factor (default value is 28).
    :param min_pixels: Minimum number of total pixels (default value is 4 * 28 * 28).
    :param max_pixels: Maximum number of total pixels (default value is 1280 * 28 * 28).
    """
    preprocess = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                height, width = image.size
                # Calculate new dimensions or transformations using a similar method
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=size_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                # Example resize
                # image = image.resize((resized_width, resized_height))
                preprocess.append(image)
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
    return preprocess


def _write_single_result(
    b_image: Image.Image,
    b_prompt: str,
    b_image_path: str,
    task_id: str,
    response: Optional[str],
    output_path: str,
    expect_output: str,
    output_suffix: str = "",
) -> Optional[Any]:
    """
    Write a single result to disk immediately when it completes.

    Args:
        b_image: The image to save
        b_prompt: The prompt used
        b_image_path: Original image path (for filename extraction)
        task_id: The task ID from the batch processor
        response: The LLM response text (may be None if failed)
        output_path: Directory to write output files
        expect_output: Expected output format ("json", "markdown", "none")
        output_suffix: Suffix to append to output filename (e.g., "_t0" for per-table)

    Returns:
        The converted result (dict for json, str for markdown) or None if failed
    """
    output_filename_base = (
        os.path.splitext(b_image_path.split("/")[-1])[0] + output_suffix
    )
    output_filename = os.path.join(output_path, f"{output_filename_base}.png")

    # Save the image
    try:
        b_image.save(output_filename)
    except Exception as e:
        logger.error(f"Failed to save image {output_filename}: {e}")

    # Handle None response (LLM call failed for this task)
    if response is None:
        logger.error(
            f"LLM response is None for task {task_id} (image: {output_filename_base}). "
            "This usually indicates an API error or timeout during inference."
        )
        return None

    # Write prompt file
    try:
        with open(f"{output_filename}_prompt.txt", "w") as f:
            f.write(b_prompt)
            f.write("\n\n\n")
            f.write(response)
    except Exception as e:
        logger.error(
            f"Failed to write prompt to file {output_filename_base}_prompt.txt: {e}"
        )

    # Check and log content type
    content_type = check_content_type(response)
    logger.info(f"Task {task_id} - Detected content type: {content_type}")
    logger.info(f"Task {task_id} - Expected content type: {expect_output}")

    if expect_output is not None and content_type != expect_output:
        logger.warning(
            f"Expected content type {expect_output} but got {content_type}. "
            "Proceeding with expected format."
        )

    # Write output based on expected format
    if expect_output == "json":
        try:
            json_dict = parse_json_markdown(response)
            with open(
                os.path.join(output_path, f"{output_filename_base}.json"), "w"
            ) as f:
                json.dump(json_dict, f, indent=4)
            logger.info(
                f"Task {task_id} - Wrote JSON result to {output_filename_base}.json"
            )
            return json_dict
        except Exception as e:
            logger.error(
                f"Failed to write response to file {output_filename_base}.json: {e}"
            )
            raise e
    elif expect_output in ("markdown", "none"):
        try:
            if expect_output == "markdown":
                output_text = parse_markdown_markdown(response)
            else:
                output_text = response
            with open(
                os.path.join(output_path, f"{output_filename_base}.md"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(output_text)
            logger.info(
                f"Task {task_id} - Wrote markdown result to {output_filename_base}.md"
            )
            return response
        except Exception as e:
            logger.error(
                f"Failed to write response to file {output_filename_base}.md: {e}"
            )
            raise e
    else:
        raise ValueError(f"Unknown expect_output type: {expect_output}")


async def process_batch(
    batch: List,
    engine: EngineLM,
    output_path: str,
    is_multimodal: bool = False,
    expect_output: str = None,
) -> list[Any] | None:
    """
    Processes a batch of images using the specified engine.

    Results are written to disk incrementally as each task completes,
    rather than waiting for all tasks to finish.

    Args:
        batch: List of (image, prompt, image_path) tuples
        engine: The LLM engine to use
        output_path: Directory to write output files
        is_multimodal: Whether to use multimodal LLM call
        expect_output: Expected output format ("json", "markdown", "none")

    Returns:
        List of converted results in original order
    """
    logger.info(f"Processing batch of images: {len(batch)}")
    logger.info(f"expect_output: {expect_output}")

    if expect_output is None:
        raise ValueError("expect_output must be specified.")

    system_prompt = SYSTEM_PROMPT
    if is_multimodal:
        llm_call = MultimodalLLMCall(engine, system_prompt=system_prompt)
    else:
        llm_call = LLMCall(engine, system_prompt=system_prompt)

    # Build mapping from batch index to metadata
    # batch is a list of tuples: (image, prompt, image_path) or (image, prompt, image_path, output_suffix)
    batch_mapping: Dict[int, Tuple[Image.Image, str, str, str]] = {}
    for i, item in enumerate(batch):
        if len(item) == 4:
            b_image, b_prompt, b_image_path, output_suffix = item
        else:
            b_image, b_prompt, b_image_path = item
            output_suffix = ""
        batch_mapping[i] = (b_image, b_prompt, b_image_path, output_suffix)

    # Thread-safe storage for results
    converted: List[Optional[Any]] = [None] * len(batch)
    write_lock = threading.Lock()
    write_errors: List[Exception] = []

    def on_result(task_id: str, response: Optional[str]) -> None:
        """
        Callback invoked when each task completes.
        Writes the result to disk immediately.
        """
        try:
            # Extract index from task_id (format: {request_id}_task_{index})
            idx = int(task_id.rsplit("_", 1)[-1])

            if idx not in batch_mapping:
                logger.error(f"Unknown task index {idx} from task_id {task_id}")
                return

            b_image, b_prompt, b_image_path, output_suffix = batch_mapping[idx]

            logger.info(f"Task {task_id} completed - Writing result immediately")

            # Write result to disk
            result = _write_single_result(
                b_image=b_image,
                b_prompt=b_prompt,
                b_image_path=b_image_path,
                task_id=task_id,
                response=response,
                output_path=output_path,
                expect_output=expect_output,
                output_suffix=output_suffix,
            )

            # Store result in thread-safe manner
            with write_lock:
                converted[idx] = result

        except Exception as e:
            logger.error(f"Error in on_result callback for task {task_id}: {e}")
            with write_lock:
                write_errors.append(e)

    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            # Prepare batch for LLM call
            if is_multimodal:
                batch_t = [[b[0], b[1]] for b in batch]
                responses = await llm_call.acall(
                    batch_t, max_tokens=4096 * 4, on_result=on_result
                )
            else:
                batch_t = [b[1] for b in batch]
                responses = await llm_call.acall(
                    batch_t, max_tokens=4096 * 4, on_result=on_result
                )

            # Check for any errors that occurred during incremental writes
            if write_errors:
                raise write_errors[0]

            # Verify responses
            assert isinstance(responses, list), "Expected a list of responses."
            assert len(responses) == len(batch), (
                f"Response count does not match the batch size. "
                f"Expected {len(batch)}, got {len(responses)}"
            )

            # Results should already be written by the callback
            # Return the converted list (populated by on_result callback)
            return converted

        except Exception as e:
            retries += 1
            logger.warning(
                f"Error processing batch (attempt {retries}/{max_retries}): {e}"
            )
            if retries >= max_retries:
                logger.error("Max retries reached. Failing the batch.")
                raise e
            else:
                logger.warning("Retrying in 2 seconds...")
                # Reset state for retry
                converted = [None] * len(batch)
                write_errors.clear()
                time.sleep(2)


def prepare_batch_with_meta(
    batched_files: list,
    frames: list[ndarray],
    doc: UnstructuredDocument,
    prompt: str,
    source_dir: str,
    context_manager: Optional["ContextProviderManager"] = None,
    units_by_file: Optional[Dict[str, "ProcessingUnit"]] = None,
) -> Generator:
    """
    Prepare batches of images with prompts for processing.

    Args:
        batched_files: List of file batches, where each batch is a list of filenames
        frames: List of image frames (numpy arrays)
        doc: The UnstructuredDocument being processed
        prompt: The prompt template
        source_dir: Directory containing the image files
        context_manager: Optional ContextProviderManager for injecting context
        units_by_file: Optional mapping from filename to ProcessingUnit for per-table processing.
                      When provided, enables per-table context injection and output suffixes.

    Yields:
        Batches of [image, prompt, image_path, output_suffix] tuples
    """
    # adding spatial context
    metadata_ocr = doc.source_metadata["ocr"]
    decorated_lines_by_page = {}

    for i, meta in enumerate(metadata_ocr):
        page_number = meta["meta"]["page"]
        lines = verbalizers("SPATIAL_FORMAT", meta)
        decorated_lines_by_page[page_number] = lines

    def decorator(text: str, line_id: int) -> str:
        """Add  line number to the text, Line ID are not necessarily row numbers"""
        # return f"{text}"
        return f"{int(line_id)} | {text}"

    debug = False
    if debug:
        for page_number in range(doc.page_count):
            print("-----------------")
            print(f"Page Number : {page_number}")
            content = doc.to_text(page_number=page_number, decorator=decorator)
            print(f"{content}")

    for batch_index, file_batch in enumerate(batched_files, start=1):
        logging.info("Processing batch %d", batch_index)
        image_paths = [os.path.join(source_dir, file_name) for file_name in file_batch]
        page_numbers = []
        prompts = []
        output_suffixes = []
        # TODO : This needs to be refactored for better readability and maintainability

        for name in image_paths:
            try:
                # Extract page number based on the filename pattern (assuming it's the last component before the extension).
                page_number = int(
                    os.path.splitext(os.path.basename(name))[0].split("_")[-1]
                )
                page_numbers.append(page_number)

                INJECTED_TEXT = ""
                injected_filename = os.path.join(
                    source_dir,
                    os.path.splitext(os.path.basename(name))[0] + "_INJECTED_TEXT.txt",
                )
                if os.path.exists(injected_filename):
                    with open(
                        injected_filename, "r", encoding="utf-8"
                    ) as injected_file:
                        # "\n" is added to separate the injected text from the OCR data
                        # also we can't do STRIP here as we have new lines in the injected alignment
                        INJECTED_TEXT = "\n" + injected_file.read()
                updated_prompt = prompt.replace("INJECTED_TEXT", INJECTED_TEXT)

                # will be used most of the time
                # content = doc.to_text(page_number=page_number - 1, decorator=decorator)
                lines_by_page = decorated_lines_by_page.get(page_number - 1, [])
                decorated_lines = []
                for line_id, line in enumerate(lines_by_page):
                    line_text = line['text']
                    decorated_line = decorator(line_text, line_id + 1)
                    decorated_lines.append(decorated_line)
                content = "\n".join(decorated_lines)

                updated_prompt = updated_prompt.replace("OCR_DATA", content).replace(
                    "OCR_TEXT", content
                )

                # Get the ProcessingUnit for this file if in per-unit mode
                file_name = os.path.basename(name)
                unit = units_by_file.get(file_name) if units_by_file else None

                # Inject context from providers (e.g., TABLE_CONTEXT_CLAIMS)
                if context_manager:
                    if unit:
                        # Per-unit mode: inject with unit context
                        updated_prompt = context_manager.inject_for_unit(
                            updated_prompt, doc, unit
                        )
                    else:
                        # Legacy mode: inject all context for page
                        updated_prompt = context_manager.inject_all(
                            updated_prompt, doc, page_number
                        )

                prompts.append(updated_prompt)
                output_suffixes.append(unit.output_suffix if unit else "")
            except ValueError:
                print(f"Unable to extract page number from filename: {name}")

        print(f"Extracted page numbers: {page_numbers}")
        # TODO : this needs to be configured via the config file
        # 2048
        images = preprocess_images_for_inference(
            image_paths, min_pixels=512 * 28 * 28, max_pixels=2048 * 28 * 28
        )
        batch_input = [
            [img, prt, img_path, suffix]
            for img, prt, img_path, suffix in zip(
                images, prompts, image_paths, output_suffixes
            )
        ]
        yield batch_input


def prepare_batch_with_meta_units(
    file_batch: List[str],
    units_by_index: Dict[int, Optional["ProcessingUnit"]],
    frames: list[ndarray],
    doc: UnstructuredDocument,
    prompt: str,
    source_dir: str,
    context_manager: Optional["ContextProviderManager"] = None,
) -> Generator:
    """
    Prepare a batch of items with prompts for processing, with explicit unit tracking.

    This function handles per-unit processing where each item in the batch
    has an associated ProcessingUnit for table-specific context injection.

    Args:
        file_batch: List of filenames in the batch
        units_by_index: Mapping from batch index to ProcessingUnit (None for legacy mode)
        frames: List of image frames (numpy arrays)
        doc: The UnstructuredDocument being processed
        prompt: The prompt template
        source_dir: Directory containing the image files
        context_manager: Optional ContextProviderManager for injecting context

    Yields:
        Batches of [image, prompt, image_path, output_suffix] tuples
    """
    # adding spatial context
    metadata_ocr = doc.source_metadata["ocr"]
    decorated_lines_by_page = {}

    for i, meta in enumerate(metadata_ocr):
        page_number = meta["meta"]["page"]
        lines = verbalizers("SPATIAL_FORMAT", meta)
        decorated_lines_by_page[page_number] = lines

    def decorator(text: str, line_id: int) -> str:
        """Add line number to the text, Line ID are not necessarily row numbers"""
        return f"{int(line_id)} | {text}"

    logging.info("Processing batch of %d items with unit tracking", len(file_batch))
    image_paths = [os.path.join(source_dir, file_name) for file_name in file_batch]
    page_numbers = []
    prompts = []
    output_suffixes = []

    for batch_idx, name in enumerate(image_paths):
        try:
            # Extract page number from filename
            page_number = int(
                os.path.splitext(os.path.basename(name))[0].split("_")[-1]
            )
            page_numbers.append(page_number)

            # Handle injected text
            INJECTED_TEXT = ""
            injected_filename = os.path.join(
                source_dir,
                os.path.splitext(os.path.basename(name))[0] + "_INJECTED_TEXT.txt",
            )
            if os.path.exists(injected_filename):
                with open(injected_filename, "r", encoding="utf-8") as injected_file:
                    INJECTED_TEXT = "\n" + injected_file.read()
            updated_prompt = prompt.replace("INJECTED_TEXT", INJECTED_TEXT)

            # Build OCR content
            lines_by_page = decorated_lines_by_page.get(page_number - 1, [])
            decorated_lines = []
            for line_id, line in enumerate(lines_by_page):
                line_text = line['text']
                decorated_line = decorator(line_text, line_id + 1)
                decorated_lines.append(decorated_line)
            content = "\n".join(decorated_lines)

            updated_prompt = updated_prompt.replace("OCR_DATA", content).replace(
                "OCR_TEXT", content
            )

            # Get the ProcessingUnit for this batch item
            unit = units_by_index.get(batch_idx)

            # Inject context from providers
            if context_manager:
                if unit:
                    # Per-unit mode: inject with unit context
                    updated_prompt = context_manager.inject_for_unit(
                        updated_prompt, doc, unit
                    )
                else:
                    # Legacy mode: inject all context for page
                    updated_prompt = context_manager.inject_all(
                        updated_prompt, doc, page_number
                    )

            prompts.append(updated_prompt)
            output_suffixes.append(unit.output_suffix if unit else "")

        except ValueError:
            logging.warning(f"Unable to extract page number from filename: {name}")

    logging.info(f"Extracted page numbers: {page_numbers}")

    # Preprocess images
    images = preprocess_images_for_inference(
        image_paths, min_pixels=512 * 28 * 28, max_pixels=2048 * 28 * 28
    )

    batch_input = [
        [img, prt, img_path, suffix]
        for img, prt, img_path, suffix in zip(
            images, prompts, image_paths, output_suffixes
        )
    ]
    yield batch_input


def scan_and_process_images(
    source_dir: str,
    output_dir: str,
    prompt: str,
    document: UnstructuredDocument,
    engine: EngineLM = None,
    is_multimodal: bool = False,
    expect_output: str = None,  # "json", "markdown", "none"
    context_manager: Optional["ContextProviderManager"] = None,
) -> None:
    """
    Synchronous wrapper for the ascan_and_process_images function.

    Scans the source directory for image files, processes each image
    with the specified prompt,
    and saves the processed outputs to the output directory.

    Parameters:
        source_dir (str): Directory containing the input image files.
        output_dir (str): Directory where processed outputs will be saved.
        prompt (str): The prompt to apply during image processing.
        document (UnstructuredDocument): The document object containing metadata and OCR data.
        engine (EngineLM): The inference engine to be used.
        is_multimodal: bool: Flag indicating if the processing is multimodal.
        expect_output (str): Expected output format ("json", "markdown", "none").
        context_manager: Optional ContextProviderManager for injecting context
                        into prompts and determining eligible pages.
    Returns:
        None
    """
    coroutine = ascan_and_process_images(
        source_dir=source_dir,
        output_dir=output_dir,
        prompt=prompt,
        document=document,
        engine=engine,
        is_multimodal=is_multimodal,
        expect_output=expect_output,
        context_manager=context_manager,
    )

    return run_async(coroutine)


@dataclass
class ProcessingItem:
    """
    Represents an item to be processed - a file with optional unit context.

    For per-unit processing, the same file may have multiple ProcessingItems
    (one per table on that page).
    """

    file_name: str
    unit: Optional["ProcessingUnit"] = None

    @property
    def output_suffix(self) -> str:
        """Get output suffix from unit if available."""
        return self.unit.output_suffix if self.unit else ""


async def ascan_and_process_images(
    source_dir: str,
    output_dir: str,
    prompt: str,
    document: UnstructuredDocument,
    engine: EngineLM = None,
    is_multimodal: bool = False,
    expect_output: Optional[str] = None,  # "json", "markdown", "none"
    context_manager: Optional["ContextProviderManager"] = None,
) -> None:
    """
    Scans the source directory for image files, processes each image
    with the specified prompt, and writes outputs to output_dir.

    When context_manager has providers, operates in per-unit mode:
    - Gets processing units (one per table) from the context manager
    - Each unit gets its own LLM call with table-specific context
    - Output files include table suffix (e.g., 00002_t0.json, 00002_t1.json)

    Parameters:
        source_dir: Directory containing the input image files.
        output_dir: Directory where processed outputs will be saved.
        prompt: The prompt to apply during image processing.
        document: The document object containing metadata and OCR data.
        engine: The inference engine to be used.
        is_multimodal: Flag indicating if the processing is multimodal.
        expect_output: Expected output format ("json", "markdown", "none").
        context_manager: Optional ContextProviderManager for injecting context
                        into prompts and determining eligible pages.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    all_files = sorted(
        f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".tif"))
    )
    if not all_files:
        logging.info("No images found in %s; nothing to do.", source_dir)
        return

    # Build a mapping from page number to filename for quick lookup
    page_to_file: Dict[int, str] = {}
    for idx, f in enumerate(all_files):
        page_num = _extract_page_number_from_filename(f)
        if page_num is None:
            page_num = idx + 1  # Fallback to 1-indexed position
        page_to_file[page_num] = f

    # Determine processing mode: per-unit (per-table) or legacy (per-page)
    processing_items: List[ProcessingItem] = []

    if context_manager and context_manager.has_providers():
        # Per-unit mode: get processing units and build item list from them
        processing_units = context_manager.get_processing_units(document)
        logging.info(
            "Per-unit processing mode: %d units from context providers",
            len(processing_units),
        )

        if not processing_units:
            logging.info("No processing units from context providers; nothing to do.")
            return

        # Build processing items from units - may contain duplicates for multi-table pages
        for unit in processing_units:
            file_name = page_to_file.get(unit.page_number)
            if file_name:
                processing_items.append(ProcessingItem(file_name=file_name, unit=unit))
            else:
                logging.warning(
                    "No file found for page %d (unit index=%s)",
                    unit.page_number,
                    unit.index,
                )

        logging.info(
            "Built %d processing items from %d units",
            len(processing_items),
            len(processing_units),
        )

        if not processing_items:
            logging.info("No processing items created; nothing to do.")
            return
    else:
        # Legacy mode: filter by eligible pages if applicable
        eligible_pages: Optional[Set[int]] = None
        if context_manager:
            eligible_pages = context_manager.get_eligible_pages(document)
            if eligible_pages:
                logging.info(
                    "Eligible pages from context providers: %s", sorted(eligible_pages)
                )

        # Build processing items from files
        for idx, f in enumerate(all_files):
            page_num = _extract_page_number_from_filename(f)
            if page_num is None:
                page_num = idx + 1

            if eligible_pages is None or page_num in eligible_pages:
                processing_items.append(ProcessingItem(file_name=f, unit=None))

        if eligible_pages is not None:
            logging.info(
                "Filtered %d files to %d based on eligible_pages",
                len(all_files),
                len(processing_items),
            )

        if not processing_items:
            logging.info("No processing items to process; nothing to do.")
            return

    # Load frames for unique files
    unique_files = list(dict.fromkeys(item.file_name for item in processing_items))
    file_to_frame = {
        f: frames_from_file(os.path.join(source_dir, f))[0] for f in unique_files
    }
    frames = [file_to_frame[item.file_name] for item in processing_items]

    mini_batch_size = 16
    batched_items = list(batchify(processing_items, mini_batch_size))
    logging.info(
        "Batching %d processing items into %d batches.",
        len(processing_items),
        len(batched_items),
    )

    async def _worker(batch: List[ProcessingItem]):
        # Extract file names and units for this batch
        file_batch = [item.file_name for item in batch]
        units_by_index = {i: item.unit for i, item in enumerate(batch)}

        gen = prepare_batch_with_meta_units(
            file_batch=file_batch,
            units_by_index=units_by_index,
            frames=frames,
            doc=document,
            prompt=prompt,
            source_dir=source_dir,
            context_manager=context_manager,
        )
        for idx, inp in enumerate(gen, 1):
            logging.debug(
                "Worker processing input #%d for batch of %d items",
                idx,
                len(batch),
            )
            await process_batch(
                inp,
                engine,
                output_dir,
                is_multimodal=is_multimodal,
                expect_output=expect_output,
            )

    tasks = [asyncio.create_task(_worker(batch)) for batch in batched_items]
    error_file_path = "/tmp/marie/llm-engine/error.log"
    if not os.path.exists(os.path.dirname(error_file_path)):
        os.makedirs(os.path.dirname(error_file_path), exist_ok=True)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, asyncio.CancelledError):
                logger.warning(f"One of the tasks was cancelled : {r}")
            elif isinstance(r, Exception):
                logger.error("Task failed:", exc_info=r)
            else:
                logger.info(f"Task completed successfully with result: {r}")

    except asyncio.CancelledError as cancel_error:
        logger.warning("One or more tasks were cancelled.")
        # DUMP FOR DEBUGGING
        with open(error_file_path, "a", encoding="utf-8") as error_file:
            error_message = f"Task(s) cancelled due to: {repr(cancel_error)}\n"
            error_file.write(error_message)
        return  # Exit early in case of cancellation

    logging.info("All image batches have been processed.")
