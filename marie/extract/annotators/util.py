import asyncio
import json
import logging
import os
import os.path
import time
from typing import Any, Generator, List, Optional

from numpy import ndarray
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

SYSTEM_PROMPT = ""  # Placeholder for the system prompt


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
    from PIL import Image

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


async def process_batch(
    batch: List,
    engine: EngineLM,
    output_path: str,
    is_multimodal: bool = False,
    expect_output: str = None,
) -> list[Any] | None:
    """
    Processes a batch of images using the specified engine.
    """
    logger.info(f"Processing batch of images : {len(batch)}")
    logger.info(f'expect_output : {expect_output}')

    if expect_output is None:
        raise ValueError("expect_output must be specified.")

    system_prompt = SYSTEM_PROMPT
    if is_multimodal:
        llm_call = MultimodalLLMCall(engine, system_prompt=system_prompt)
    else:
        llm_call = LLMCall(engine, system_prompt=system_prompt)

    max_retries = 3
    retries = 0
    # FIXME: The tokens limit is set to 4096 * 4, which is quite high, need to make this configurable per model

    while retries < max_retries:
        try:
            # extract the image and prompt from the batch which is a list of tuples(image, prompt, image_path)
            # responses = llm_call(batch_t, max_tokens=2048, guided_regex=GUIDED_REGEX_SEGMENT, guided_backend='outlines')
            batch_t = [
                [b[0], b[1]] for b in batch
            ]  # batch_t is a list of tuples(image, prompt, image_path) but we only need image and prompt
            if is_multimodal:
                responses = await llm_call.acall(batch_t, max_tokens=4096 * 4)
            else:
                batch_t = [b[1] for b in batch]
                responses = await llm_call.acall(
                    batch_t,
                    max_tokens=4096 * 4,
                )

            assert isinstance(responses, list), "Expected a list of responses."
            assert len(responses) == len(
                batch
            ), f"Response count does not match the batch size. Expected {len(batch)}, got {len(responses)}"

            converted = []
            for i, (b_image, b_prompt, b_image_path) in enumerate(batch):
                logger.info("-----------  Processing image ------------")
                logger.info(f"Processing image : {i}")
                # response = responses[i]
                task_id = responses[i][0]
                response = responses[i][1]

                output_filename_base = os.path.splitext(b_image_path.split("/")[-1])[0]
                output_filename = os.path.join(
                    output_path, f"{output_filename_base}.png"
                )
                b_image.save(output_filename)

                # Handle None response (LLM call failed for this task)
                if response is None:
                    logger.error(
                        f"LLM response is None for task {task_id} (image {i}: {output_filename_base}). "
                        "This usually indicates an API error or timeout during inference."
                    )
                    converted.append(None)
                    continue

                try:
                    with open(f"{output_filename}_prompt.txt", "w") as f:
                        f.write(b_prompt)
                        f.write("\n\n\n")
                        f.write(response)
                except Exception as e:
                    logger.error(
                        f"Failed to write prompt to file {output_filename_base}_prompt.txt. : {e}"
                    )

                content_type = check_content_type(response)
                logger.info(f"Detected content type : {content_type}")
                logger.info(f"Expected content type : {expect_output}")

                if expect_output is not None and content_type != expect_output:
                    logger.warning(
                        f"Expected content type {expect_output} but got {content_type}. Skipping this response."
                    )

                if expect_output == "json":
                    try:
                        with open(
                            os.path.join(output_path, f"{output_filename_base}.json"),
                            "w",
                        ) as f:
                            json_dict = parse_json_markdown(response)
                            json.dump(json_dict, f, indent=4)

                            converted.append(json_dict)
                    except Exception as e:
                        print(
                            f"Failed to write response to file {output_filename_base}.json. : {e}"
                        )
                        raise e
                elif (
                    expect_output == "markdown" or expect_output == "none"
                ):  # some responses are just plain markdown without any ```markdown``` block
                    try:
                        with open(
                            os.path.join(output_path, f"{output_filename_base}.md"),
                            "w",
                            encoding='utf-8',
                        ) as f:
                            if expect_output == "markdown":
                                output_text = parse_markdown_markdown(response)
                            else:
                                output_text = response
                            f.write(output_text)
                            converted.append(response)
                    except Exception as e:
                        logger.error(
                            f"Failed to write response to file {output_filename_base}.md. : {e}"
                        )
                        raise e
                else:
                    raise Exception(
                        f"Unknown content type {content_type} for response = {response}"
                    )
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
                time.sleep(2)


def prepare_batch_with_meta(
    batched_files: list,
    frames: list[ndarray],
    doc: UnstructuredDocument,
    prompt: str,
    source_dir: str,
) -> Generator:
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
                prompts.append(updated_prompt)
            except ValueError:
                print(f"Unable to extract page number from filename: {name}")
        print(f"Extracted page numbers: {page_numbers}")
        # TODO : this needs to be configured via the config file
        # 2048
        images = preprocess_images_for_inference(
            image_paths, min_pixels=512 * 28 * 28, max_pixels=2048 * 28 * 28
        )
        batch_input = [
            [img, prt, img_path]
            for img, prt, img_path in zip(images, prompts, image_paths)
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
    )

    return run_async(coroutine)


async def ascan_and_process_images(
    source_dir: str,
    output_dir: str,
    prompt: str,
    document: UnstructuredDocument,
    engine: EngineLM = None,
    is_multimodal: bool = False,
    expect_output: Optional[str] = None,  # "json", "markdown", "none"
) -> None:
    """
    Scans the source directory for image files, processes each image
    with the specified prompt, and writes outputs to output_dir.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    files = sorted(
        f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".tif"))
    )
    if not files:
        logging.info("No images found in %s; nothing to do.", source_dir)
        return

    frames = [frames_from_file(os.path.join(source_dir, f))[0] for f in files]
    mini_batch_size = 16
    batched_files = list(batchify(files, mini_batch_size))
    logging.info("Batching %d images into %d batches.", len(files), len(batched_files))

    async def _worker(batch):
        gen = prepare_batch_with_meta([batch], frames, document, prompt, source_dir)
        for idx, inp in enumerate(gen, 1):
            logging.debug(
                "Worker processing input #%d for batch of %d files",
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

    tasks = [asyncio.create_task(_worker(batch)) for batch in batched_files]
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
