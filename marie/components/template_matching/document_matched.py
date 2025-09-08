import base64
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from marie.components.template_matching import (
    BaseTemplateMatcher,
    CompositeTemplateMatcher,
    MetaTemplateMatcher,
    VQNNFTemplateMatcher,
)
from marie.components.template_matching.model import (
    TemplateMatchingRequestDoc,
    TemplateSelector,
)
from marie.ocr import DefaultOcrEngine, OcrEngine
from marie.utils.json import load_json_file, store_json_object
from marie.utils.resize_image import resize_image
from marie.utils.utils import ensure_exists

# Cache for convert_template_selectors results
_CTS_CACHE: dict[
    str, tuple[list[np.ndarray], list[list[int]], list[str], list[str]]
] = {}
_CTS_CACHE_LOCK = Lock()


def convert_template_selectorsV1(
    selectors: List[TemplateSelector],
    window_size: Union[List[int], tuple[int, int]],
):
    """
    Convert TemplateSelector to Template Matching Selectors
    :param selectors:
    :param window_size:
    :return:
    """

    print(f"Converting {len(selectors)} selector(s)")
    template_frames = []
    template_bboxes = []
    template_labels = []
    template_texts = []

    for i, selector in enumerate(selectors):
        buf = io.BytesIO(base64.b64decode(selector.frame))
        image = Image.open(buf)
        image = image.convert("RGB")
        frame = np.array(image)
        frame = frame[:, :, ::-1].copy()

        boxes_xywh = [
            selector.bbox
        ]  # currently only one bbox is supported per selector
        region = selector.region
        label = selector.label
        text = selector.text

        if selector.create_window:
            frame, coord = resize_image(
                frame,
                window_size,
                keep_max_size=True,
            )
            boxes_xywh = [coord]

        (
            sel_template_frames,
            sel_template_bboxes,
        ) = BaseTemplateMatcher.extract_windows(
            frame, boxes_xywh, window_size, allow_padding=True
        )

        template_frames.extend(sel_template_frames)
        template_bboxes.extend(sel_template_bboxes)
        template_labels.append(label)
        template_texts.append(text)
        assert (
            len(template_frames)
            == len(template_bboxes)
            == len(template_labels)
            == len(template_texts)
        )
        # Converting 351 selector(s)
        ensure_exists("/tmp/dim/template")
        for template_frame in template_frames:
            cv2.imwrite(
                f"/tmp/dim/template/template_frame_SELECTOR_{i}.png", template_frame
            )

    return template_frames, template_bboxes, template_labels, template_texts


def _selectors_cache_key(
    selectors: List[TemplateSelector], window_size: Union[List[int], tuple[int, int]]
) -> str:
    # Build a stable hash from selector-defining fields and frame content
    normalized_ws = list(window_size) if isinstance(window_size, tuple) else window_size
    items = []
    for s in selectors:
        # s.frame is base64 string; hash it to avoid storing raw image in the key
        frame_hash = hashlib.sha256(s.frame.encode("utf-8")).hexdigest()
        items.append(
            {
                "bbox": list(s.bbox),
                "label": s.label,
                "text": s.text,
                "create_window": bool(s.create_window),
                "top_k": int(s.top_k),
                "frame_hash": frame_hash,
                # region may be None or list[int]; normalize
                "region": None if s.region is None else list(s.region),
            }
        )
    payload = {"selectors": items, "window_size": list(normalized_ws)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def convert_template_selectors(
    selectors: List[TemplateSelector],
    window_size: Union[List[int], tuple[int, int]],
):
    """
    Convert TemplateSelector to Template Matching Selectors
    :param selectors:
    :param window_size:
    :return:
    """

    print(f"Converting {len(selectors)} selector(s)")
    start_time = time.time()
    # Check cache
    cache_key = _selectors_cache_key(selectors, window_size)
    with _CTS_CACHE_LOCK:
        cached = _CTS_CACHE.get(cache_key)
    if cached is not None:
        # Return shallow copies to avoid accidental external mutations of cached lists
        tf, tb, tl, tt = cached
        return list(tf), list(tb), list(tl), list(tt)

    def process_selector(args):
        i, selector = args
        buf = io.BytesIO(base64.b64decode(selector.frame))
        image = Image.open(buf)
        image = image.convert("RGB")
        frame = np.array(image)
        frame = frame[:, :, ::-1].copy()

        boxes_xywh = [
            selector.bbox
        ]  # currently only one bbox is supported per selector
        label = selector.label
        text = selector.text

        if selector.create_window:
            frame, coord = resize_image(
                frame,
                window_size,
                keep_max_size=True,
            )
            boxes_xywh = [coord]

        (
            sel_template_frames,
            sel_template_bboxes,
        ) = BaseTemplateMatcher.extract_windows(
            frame, boxes_xywh, window_size, allow_padding=True
        )

        if False:
            ensure_exists("/tmp/dim/template")
            for template_frame in sel_template_frames:
                cv2.imwrite(
                    f"/tmp/dim/template/template_frame_SELECTOR_{i}.png", template_frame
                )
        return sel_template_frames, sel_template_bboxes, label, text

    template_frames = []
    template_bboxes = []
    template_labels = []
    template_texts = []

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, cpu_count * 2 // 3)  # Use two-thirds of available CPU cores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_selector, enumerate(selectors))
        for sel_template_frames, sel_template_bboxes, label, text in results:
            template_frames.extend(sel_template_frames)
            template_bboxes.extend(sel_template_bboxes)
            template_labels.append(label)
            template_texts.append(text)

    # Store in cache
    with _CTS_CACHE_LOCK:
        _CTS_CACHE[cache_key] = (
            list(template_frames),
            list(template_bboxes),
            list(template_labels),
            list(template_texts),
        )

    end_time = time.time()
    print(f"convert_template_selectors took {end_time - start_time:.2f}) seconds")
    return template_frames, template_bboxes, template_labels, template_texts


def convert_template_selectorsV2(
    selectors: List[TemplateSelector],
    window_size: Union[List[int], tuple[int, int]],
):
    """
    Convert TemplateSelector to Template Matching Selectors
    :param selectors:
    :param window_size:
    :return:
    """

    print(f"Converting {len(selectors)} selector(s)")

    def process_selector(args):
        i, selector = args
        buf = io.BytesIO(base64.b64decode(selector.frame))
        image = Image.open(buf)
        image = image.convert("RGB")
        frame = np.array(image)
        frame = frame[:, :, ::-1].copy()

        boxes_xywh = [
            selector.bbox
        ]  # currently only one bbox is supported per selector
        label = selector.label
        text = selector.text

        if selector.create_window:
            frame, coord = resize_image(
                frame,
                window_size,
                keep_max_size=True,
            )
            boxes_xywh = [coord]

        (
            sel_template_frames,
            sel_template_bboxes,
        ) = BaseTemplateMatcher.extract_windows(
            frame, boxes_xywh, window_size, allow_padding=True
        )

        if False:
            ensure_exists("/tmp/dim/template")
            for template_frame in sel_template_frames:
                cv2.imwrite(
                    f"/tmp/dim/template/template_frame_SELECTOR_{i}.png", template_frame
                )
        return sel_template_frames, sel_template_bboxes, label, text

    template_frames = []
    template_bboxes = []
    template_labels = []
    template_texts = []

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, cpu_count * 2 // 3)  # Use two-thirds of available CPU cores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_selector, enumerate(selectors))
        for sel_template_frames, sel_template_bboxes, label, text in results:
            template_frames.extend(sel_template_frames)
            template_bboxes.extend(sel_template_bboxes)
            template_labels.append(label)
            template_texts.append(text)

    return template_frames, template_bboxes, template_labels, template_texts


def load_template_matching_definitions(
    definition_file: str,
) -> TemplateMatchingRequestDoc:
    """
    Load the template matching definitions from the template matching definition file.
    """
    data = load_json_file(definition_file)
    template_matching_request = TemplateMatchingRequestDoc(
        asset_key="TEMPLATE",
        pages=[],
        id='',
        score_threshold=0.8,
        scoring_strategy="weighted",
        max_overlap=0.5,
        window_size=[512, 512],
        matcher="composite",
        downscale_factor=1,
        selectors=[],
    )

    for selector in data["selectors"]:
        print(f"Adding selector: {selector}")
        if "region" not in selector:
            selector["region"] = None

        template_matching_request.selectors.append(
            TemplateSelector(
                region=selector["region"],
                frame=selector["frame"],
                bbox=selector["bbox"],
                label=selector["label"],
                text=selector["text"],
                create_window=selector["create_window"],
                top_k=selector["top_k"],
            )
        )

    # print information about the loaded template matching definitions
    print(f"Loaded template matching definitions from {definition_file}")
    print(f"Template matching request: {template_matching_request}")

    return template_matching_request


def get_template_matchers():
    matcher_vqnnft = VQNNFTemplateMatcher(model_name_or_path="NONE")
    matcher_meta = MetaTemplateMatcher(model_name_or_path="NONE")
    matcher = CompositeTemplateMatcher(matchers=[matcher_vqnnft, matcher_meta])

    return matcher, matcher_meta, matcher_vqnnft


def get_ocr_engine() -> OcrEngine:
    """Get the OCR engine"""
    use_cuda = torch.cuda.is_available()
    ocr_engine = DefaultOcrEngine(cuda=use_cuda)

    return ocr_engine


def get_matchers():
    matcher_vqnnft = VQNNFTemplateMatcher(model_name_or_path="NONE")
    matcher_meta = MetaTemplateMatcher(model_name_or_path="NONE")
    matcher = CompositeTemplateMatcher(
        matchers=[matcher_meta, matcher_vqnnft], break_on_match=True
    )

    return matcher, matcher_meta, matcher_vqnnft


def match_templates(
    frames: list[np.ndarray],
    definition: TemplateMatchingRequestDoc,
    matcher: BaseTemplateMatcher,
    ocr_results: dict,
):
    """
    Match the templates in the frames using the template matching definition.
    """
    print(
        f"Matching templates in the frames using the template matching definition: {definition}"
    )
    print(f"Frames: {len(frames)}")
    doc = definition
    if len(doc.selectors) == 0:
        return {"error": "selectors not present"}

    (
        template_frames,
        template_bboxes,
        template_labels,
        template_texts,
    ) = convert_template_selectors(doc.selectors, doc.window_size)

    results = matcher.run(
        frames=frames,
        # TODO: convert to Pydantic model
        template_frames=template_frames,
        template_boxes=template_bboxes,
        template_labels=template_labels,
        template_texts=template_texts,
        metadata=ocr_results,
        score_threshold=0.9,  # doc.score_threshold,
        scoring_strategy='weighted',  # doc.scoring_strategy,
        max_overlap=0.5,
        max_objects=2,
        window_size=(doc.window_size[0], doc.window_size[1]),
        downscale_factor=doc.downscale_factor,
    )

    # group the results by the frame_index
    results_dict = {}
    for result in results:
        key = f"{result.frame_index}"
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(result)

    store_json_object(ocr_results, "/tmp/dim/metadata-raw.json")

    print("Results:")
    print(results)

    metadata = {
        "ocr_results": ocr_results,
        "template_matching": results_dict,
    }
    store_json_object(metadata, "/tmp/dim/metadata.json")

    return results_dict
