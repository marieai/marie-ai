import tempfile
from os import PathLike
from typing import Union

import numpy as np

from marie.boxes import BoxProcessorUlimDit
from marie.document import TrOcrProcessor
from marie.executor.util import setup_cache
from marie.logging_core.predefined import default_logger as logger
from marie.ocr import DefaultOcrEngine, MockOcrEngine, OcrEngine, VotingOcrEngine
from marie.renderer import TextRenderer
from marie.utils.json import load_json_file
from marie.utils.utils import ensure_exists


def get_words_and_boxes(
    ocr_results, page_index: int, include_lines: bool = False
) -> tuple[list[str], list[list[int]]] | tuple[list[str], list[list[int]], list[int]]:
    """
    Get words and boxes from OCR results.
    :param ocr_results: OCR results
    :param page_index: Page index to get words and boxes from.
    :param include_lines: Include lines in the result.
    :return:
    """
    words = []
    boxes = []
    lines = []
    if not ocr_results:
        return words, boxes
    if page_index >= len(ocr_results):
        raise ValueError(f"Page index {page_index} is out of range.")

    for w in ocr_results[page_index]["words"]:
        boxes.append(w["box"])
        words.append(w["text"])
        lines.append(w["line"])
    if include_lines:
        return words, boxes, lines
    return words, boxes


def meta_to_text(
    meta_or_path: Union[dict | str | PathLike], text_output_path: str = None
) -> str:
    """
    Convert meta data to text.

    :param meta_or_path: Meta data or path to meta data.
    :param text_output_path:  Path to text output file. If not provided, a temporary file will be used.
    :return:
    """

    if isinstance(meta_or_path, (str, PathLike)):
        results = load_json_file(meta_or_path)
    else:
        results = meta_or_path

    # create a fake frames array from metadata in the results, this is needed for the renderer for sizing
    frames = []

    for result in results:
        meta = result["meta"]["imageSize"]
        width = meta["width"]
        height = meta["height"]
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    # write to temp file and read it back
    if text_output_path:
        tmp_file = open(text_output_path, "w")
    else:
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")

    with open(tmp_file.name, "w", encoding="utf-8") as f:
        for result in results:
            lines = result["lines"]
            lines = sorted(lines, key=lambda k: k["line"])
            for i, line in enumerate(lines):
                f.write(line["text"])
                if i < len(lines) - 1:
                    f.write("\n")
    tmp_file.close()

    with open(tmp_file.name, "r") as f:
        return f.read()


def get_known_ocr_engines(
    device: str = "cuda", engine: str = None
) -> dict[str, OcrEngine]:
    """
    Get the known OCR engines
    mock : Mock OCR engine, returns dummy results
    default : Default OCR engine, uses the best OCR engine available on the system
    best : Voting OCR engine, uses ensemble of OCR engines to perform OCR on the document

    Most GPU will not have enough memory to run multiple OCR engines in parallel and hence it is recommended to use
    the default OCR engine on GPU. If you have a large GPU with enough memory, you can use the best OCR engine.

    :param device: device to use for OCR (cpu or cuda)
    :param engine: engine to use for OCR (mock, default, best)
    :return: OCR engines
    """

    use_cuda = False
    if device == "cuda":
        use_cuda = True

    logger.info(f"Getting OCR engine using engine : {engine}, device : {device}")
    setup_cache(list_of_models=None)

    box_processor = BoxProcessorUlimDit(
        work_dir=ensure_exists("/tmp/boxes"),
        cuda=use_cuda,
    )

    trocr_processor = TrOcrProcessor(work_dir=ensure_exists("/tmp/icr"), cuda=use_cuda)

    ocr_engines = dict()

    if engine is None:
        ocr_engines["mock"] = MockOcrEngine(cuda=use_cuda, box_processor=box_processor)
        ocr_engines["default"] = DefaultOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
        ocr_engines["best"] = VotingOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
    elif engine == "mock":
        ocr_engines["mock"] = MockOcrEngine(cuda=use_cuda, box_processor=box_processor)
    elif engine == "default":
        ocr_engines["default"] = DefaultOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
    elif engine == "best":
        ocr_engines["best"] = VotingOcrEngine(
            cuda=use_cuda,
            box_processor=box_processor,
            default_ocr_processor=trocr_processor,
        )
    else:
        raise ValueError(f"Invalid OCR engine : {engine}")

    return ocr_engines
