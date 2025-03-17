import json

import cv2
import numpy as np
import streamlit as st
import torch
from canvas_util import ImageUtils
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from util import generate_unique_key, process_image

from marie.boxes import BoxProcessorUlimDit
from marie.components.document_indexer.transformers_seq2seq import (
    TransformersSeq2SeqDocumentIndexer,
)
from marie.document import TrOcrProcessor
from marie.ocr.util import get_words_and_boxes
from marie.utils.docs import docs_from_image

use_cuda = torch.cuda.is_available()

if "extracted" not in st.session_state:
    st.session_state.extracted = {
        "snippet": None,
        "bboxes_img": None,
        "overlay_image": None,
        "lines_img": None,
        "result_json": None,
        "extracted_text": "zero",
    }

if "text_prompt" not in st.session_state:
    st.session_state.text_prompt = None


@st.cache_resource
def build_ocr_engine():
    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )
    icr_processor = TrOcrProcessor(
        models_dir="/mnt/data/marie-ai/model_zoo/trocr", cuda=use_cuda
    )
    return box_processor, icr_processor


@st.cache_resource
def build_indexer():
    model_name_or_path = "marie/key-value-relation"
    resolved_model_path = model_name_or_path
    return TransformersSeq2SeqDocumentIndexer(
        model_name_or_path=resolved_model_path, ocr_engine=None
    )


def generate_output(prompt, snippet, words, boxes):
    if isinstance(snippet, np.ndarray):
        snippet = Image.fromarray(snippet)
    documents = docs_from_image(snippet)
    indexer = build_indexer()

    print("documents", len(documents))
    print("words", len(words))
    print("boxes", len(boxes))

    results = indexer.run(
        documents=documents, words=[words], boxes=[boxes], prompts=[prompt]
    )

    for document in results:
        print("results", document.tags)
        indexer_result = document.tags['indexer']
        kv = indexer_result['kv']
        print("indexer", kv)
        return kv


def main():
    st.set_page_config(
        page_title="Relation Extraction Toolbox",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            #MainMenu, header, footer {visibility: hidden;}

            div[class^='block-container'] { padding-top: 0.5rem; }

            .st-emotion-cache-16txtl3{
                padding: 2.7rem 0.6rem
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # st.write(st.session_state)
    utils = ImageUtils()

    st.sidebar.header("Upload Section")
    uploaded_image = st.sidebar.file_uploader(
        "Upload an Image (TIFF/PDF)",
        type=["png", "jpg", "jpeg", "tiff", "pdf"],
        key="source",
    )

    submit = st.sidebar.button("Submit")
    canvas_result = None

    col1, col2 = st.columns(2)
    with st.container(border=True):
        with col1:
            mode = "rect"
            if uploaded_image is not None:
                raw_image, raw_size = utils.read_image(uploaded_image, False)
                resized_image, resized_size = utils.resize_image_for_canvas(raw_image)
                canvas_result = utils.get_canvas(
                    resized_image, key="canvas-source", update_streamlit=True, mode=mode
                )

                st.write(f"Uploaded filename: `{uploaded_image.name}`")
                unique_key = generate_unique_key(uploaded_image.name)
                st.write(f"Unique key: `{unique_key}`")
                st.session_state.unique_key = unique_key

        with col2:
            if submit:
                resized_boxes = utils.get_resized_boxes(canvas_result)
                # left_upper point and right_lower point : [x1, y1, x2, y2]
                raw_boxes = utils.get_raw_boxes(resized_boxes, raw_size, resized_size)

                if len(raw_boxes) == 0:
                    st.warning("No selectors defined", icon="⚠️")
                    return  # stop the execution

                frame_src = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
                raw_boxes_xywh = [
                    [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    for box in raw_boxes
                ]

                # we only process the first box
                box = raw_boxes_xywh[0]
                x, y, w, h = box
                snippet = frame_src[y : y + h, x : x + w, :]
                box_processor, ocr_processor = build_ocr_engine()
                bboxes_img, overlay_image, lines_img, result_json, extracted_text = (
                    process_image(snippet, box_processor, ocr_processor)
                )

                st.session_state.extracted = {
                    "snippet": snippet,
                    "bboxes_img": bboxes_img,
                    "overlay_image": overlay_image,
                    "lines_img": lines_img,
                    "result_json": result_json,
                    "extracted_text": extracted_text,
                }

            if st.session_state.extracted["snippet"] is not None:
                snippet = st.session_state.extracted["snippet"]
                extracted_text = st.session_state.extracted["extracted_text"]

                st_snippet = st.image(
                    snippet, caption="Snippet", use_container_width=False
                )
                st_extracted = st.text_area(
                    "Extracted Text", value=extracted_text, height=400, max_chars=None
                )

            st.markdown('---')
            augment = st.button("Extract")

            if augment:
                st.write("Extracting...")
                if st.session_state.extracted["snippet"] is not None:
                    snippet = st.session_state.extracted["snippet"]

                if st.session_state.extracted["result_json"] is not None:
                    ocr_results = st.session_state.extracted["result_json"]
                    ocr_results = json.loads(ocr_results)

                prompt = st.session_state.text_prompt
                words, boxes = get_words_and_boxes(ocr_results, 0)
                print('-' * 50)

                converted_output = generate_output(prompt, snippet, words, boxes)

                result_json = json.loads(st.session_state.extracted["result_json"])
                result = {
                    'query': prompt,
                    'result': converted_output,
                    'metadata': result_json,
                }

                # save_to_json_file(result, snippet, output_filename)
                st.write(converted_output)

    st.markdown('---')

    col1, col2 = st.columns(2)
    with st.container(border=True):
        with col1:
            extracted_text = "Default Prompt Text"
            if st.session_state.extracted["extracted_text"] is not None:
                extracted_text = st.session_state.extracted["extracted_text"]

            env = Environment(loader=FileSystemLoader("templates"))
            template = env.get_template("inference_prompt.txt.j2")
            default_prompt = template.render(text=extracted_text)
            st.header("Prompt")
            user_input = st.text_area("Prompt", value=default_prompt, height=500)
            st.session_state.text_prompt = user_input

        with col2:
            st.header("Params")


if __name__ == "__main__":
    main()
