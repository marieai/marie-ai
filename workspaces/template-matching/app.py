import base64
import json
import logging
import os
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from marie import __version__ as marie_version
from marie.boxes.box_processor import PSMode
from marie.components.template_matching.composite_template_maching import (
    CompositeTemplateMatcher,
)
from marie.components.template_matching.meta_template_matching import (
    MetaTemplateMatcher,
)
from marie.ocr import CoordinateFormat, DefaultOcrEngine, OcrEngine
from marie.ocr.util import meta_to_text
from marie.utils.resize_image import resize_image

print("marie_version", marie_version)

from marie.components.template_matching.vqnnf_template_matching import (
    BaseTemplateMatcher,
    VQNNFTemplateMatcher,
)


@st.cache_resource
def get_template_matchers():
    matcher_vqnnft = VQNNFTemplateMatcher(model_name_or_path="NONE")
    matcher_meta = MetaTemplateMatcher(model_name_or_path="NONE")
    matcher = CompositeTemplateMatcher(matchers=[matcher_vqnnft, matcher_meta])

    return matcher, matcher_meta, matcher_vqnnft


class ImageUtils:
    def __init__(self):
        pass

    def read_image(self, uploaded_image):
        """Read the uploaded image"""
        raw_image = Image.open(uploaded_image).convert("RGB")
        width, height = raw_image.size
        return raw_image, (width, height)

    def resize_image(self, raw_image, square=960):
        """Resize the mask so it fits inside a 544x544 square"""
        width, height = raw_image.size
        # return raw_image.resize((width, height)), (width, height)

        return raw_image.resize((square, square)), (square, square)

    def get_canvas(
        self, resized_image, key="canvas", update_streamlit=True, mode="rect"
    ):
        """Retrieves the canvas to receive the bounding boxes
        Args:
        resized_image(Image.Image): the resized uploaded image
        key(str): the key to initiate the canvas component in streamlit
        """
        width, height = resized_image.size

        canvas_result = st_canvas(
            # fill_color="rgba(255,0, 0, 0.1)",
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=2,
            stroke_color="rgba(255,0,0,1)",
            background_color="rgba(0,0,0,1)",
            background_image=resized_image,
            update_streamlit=update_streamlit,
            height=960,
            width=960,
            drawing_mode=mode,
            key=key,
        )
        return canvas_result

    def get_resized_boxes(self, canvas_result):
        """Get the resized boxes from the canvas result"""
        objects = canvas_result.json_data["objects"]
        resized_boxes = []
        for obj in objects:
            print("object", obj)
            left, top = int(obj["left"]), int(obj["top"])  # upper left corner
            width, height = int(obj["width"]), int(
                obj["height"]
            )  # box width and height
            right, bottom = left + width, top + height  # lower right corner
            resized_boxes.append([left, top, right, bottom])
        return resized_boxes

    def get_raw_boxes(self, resized_boxes, raw_size, resized_size):
        """Convert the resized boxes to raw boxes"""

        raw_width, raw_height = raw_size
        resized_width, resized_height = resized_size
        raw_boxes = []

        for box in resized_boxes:
            left, top, right, bottom = box
            raw_left = int(left * raw_width / resized_width)
            raw_top = int(top * raw_height / resized_height)
            raw_right = int(right * raw_width / resized_width)
            raw_bottom = int(bottom * raw_height / resized_height)
            raw_boxes.append([raw_left, raw_top, raw_right, raw_bottom])
        return raw_boxes


@st.cache_resource
def get_ocr_engine() -> OcrEngine:
    """Get the OCR engine"""
    use_cuda = torch.cuda.is_available()
    ocr_engine = DefaultOcrEngine(cuda=use_cuda)

    return ocr_engine


def main():
    st.set_page_config(
        page_title="Marie-AI Template Matching",
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

    utils = ImageUtils()

    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s%(levelname)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    st.sidebar.write("Matching parameters X")
    scol1, scol2 = st.sidebar.columns([5, 5])

    with scol1:
        max_matches_number = st.number_input(
            "Max matches", min_value=1, max_value=10, value=1, step=1, format="%d"
        )
    with scol2:
        score_threshold_number = st.number_input(
            "Match threshold",
            min_value=1,
            max_value=100,
            value=40,
            step=1,
            format="%d",
        )

    scol1, scol2 = st.sidebar.columns([5, 5])
    with scol1:
        window_size_h = st.number_input(
            "Window size height(px)",
            min_value=128,
            max_value=512,
            value=256,
            step=1,
            format="%d",
        )
    with scol2:
        window_size_w = st.number_input(
            "Window size width(px)",
            min_value=128,
            max_value=512,
            value=256,
            step=1,
            format="%d",
        )
    st.sidebar.divider()

    mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"

    reinforce_mode = (
        True if st.sidebar.checkbox("Reinforce mode(OCR) ", False) else False
    )

    matching_mode = st.sidebar.radio(
        "Matching mode:", ("Composite", "VQNNF", "Meta"), index=0
    )

    uploaded_image = st.sidebar.file_uploader(
        "Upload template source: ",
        type=["jpg", "jpeg", "png", "webp"],
        key="source",
    )
    uploaded_target = st.sidebar.file_uploader(
        "Upload document to match: ",
        type=["jpg", "jpeg", "png", "webp"],
        key="target",
    )

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    submit = st.sidebar.button("Submit")

    canvas_result = None
    col1, col2 = st.columns(2)
    with st.container(border=True):
        with col1:
            # st.header("Source image")
            if uploaded_image is not None:
                raw_image, raw_size = utils.read_image(uploaded_image)
                resized_image, resized_size = utils.resize_image(raw_image)
                # read bbox input
                canvas_result = utils.get_canvas(
                    resized_image, key="canvas-source", update_streamlit=True, mode=mode
                )
        with col2:
            # st.header("Matching target")
            if uploaded_target is not None:
                raw_image_target, raw_size_target = utils.read_image(uploaded_target)
                resized_image_target, resized_size_target = utils.resize_image(
                    raw_image_target
                )
                canvas_output = utils.get_canvas(
                    resized_image_target,
                    key="canvas-target",
                    update_streamlit=True,
                )

            if False:
                if canvas_result is not None:
                    # if canvas_result.image_data is not None:
                    #     st.image(canvas_result.image_data)
                    if canvas_result.json_data is not None:
                        objects = pd.json_normalize(
                            canvas_result.json_data["objects"]
                        )  # need to convert obj to str because PyArrow
                        for col in objects.select_dtypes(include=["object"]).columns:
                            objects[col] = objects[col].astype("str")
                        st.dataframe(objects)

    with st.container():
        if submit:
            ocr_results = None
            raw_image = raw_image.convert("RGB")
            frame = np.array(raw_image)

            resized_boxes = utils.get_resized_boxes(canvas_result)
            # left_upper point and right_lower point : [x1, y1, x2, y2]
            raw_boxes = utils.get_raw_boxes(resized_boxes, raw_size, resized_size)

            if len(raw_boxes) == 0:
                st.warning("No selectors defined", icon="⚠️")
                return  # stop the execution

            raw_boxes_xywh = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in raw_boxes
            ]

            st.write(f"Reinforce mode : {reinforce_mode}")
            raw_text = ["" for _ in raw_boxes_xywh]

            if reinforce_mode:
                ocr_engine = get_ocr_engine()
                ocr_results = ocr_engine.extract(
                    [frame], PSMode.SPARSE, CoordinateFormat.XYWH
                )

                raw_text = []
                for box in raw_boxes_xywh:
                    x, y, w, h = box
                    snippet = frame[y : y + h, x : x + w, :]
                    snippet_result = ocr_engine.extract(
                        [snippet], PSMode.SPARSE, CoordinateFormat.XYWH
                    )

                    snippet_txt = meta_to_text(snippet_result)
                    snippet_txt = snippet_txt.replace("\n", " ")
                    snippet_txt = " ".join(snippet_txt.split())
                    raw_text.append(snippet_txt)

            st.write("Boxes - original/converted")
            max_width = max([box[2] - box[0] for box in raw_boxes])
            column_configuration = {
                "snippet": st.column_config.ImageColumn(
                    "snippet", help="Document snippet", width=max_width
                ),
            }

            def image_to_base64(img) -> str:
                with BytesIO() as buffer:
                    img.save(buffer, "png")  # or 'jpeg'
                    return base64.b64encode(buffer.getvalue()).decode()

            def image_formatter(img) -> str:
                return f"data:image/png;base64,{image_to_base64(img)}"

            os.makedirs("/tmp/template", exist_ok=True)
            rows = []

            for i, (s, result, z, text) in enumerate(
                zip(resized_boxes, raw_boxes, raw_boxes_xywh, raw_text)
            ):
                snippet = raw_image.crop(result)
                snippet.save(f"/tmp/template/snippet_{i}.png")

                rows.append(
                    {
                        "snippet": image_formatter(snippet),
                        "reinforced-text": text,
                        "sx1": s[0],
                        "sy1": s[1],
                        "sx2": s[2],
                        "sy2": s[3],
                        "x1": result[0],
                        "y1": result[1],
                        "x2": result[2],
                        "y2": result[3],
                        "x": z[0],
                        "y": z[1],
                        "w": z[2],
                        "h": z[3],
                    }
                )

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                column_config=column_configuration,
            )

            st.write("Received the following sample:")
            st.write("Matching mode: ", matching_mode)

            matcher_composite, matcher_meta, matcher_vqnnft = get_template_matchers()
            matcher = matcher_composite

            if matching_mode == "VQNNF":
                matcher = matcher_vqnnft
            elif matching_mode == "Meta":
                matcher = matcher_meta

            window_size = (512, 512)

            template_frames = []
            template_bboxes = []
            template_labels = []
            template_texts = []

            for i, (s, result, z, text) in enumerate(
                zip(resized_boxes, raw_boxes, raw_boxes_xywh, raw_text)
            ):
                x, y, w, h = z
                template = frame[y : y + h, x : x + w, :]
                cv2.imwrite(f"/tmp/template/template_frame_{i}.png", template)

                template, coord = resize_image(
                    template,
                    desired_size=window_size,
                    color=(255, 255, 255),
                    keep_max_size=True,
                )

                cv2.imwrite(f"/tmp/dim/template/template_{i}.png", template)

                template_frames.append(template)
                template_bboxes.append(coord)
                template_labels.append(f"label_{i}")
                template_texts.append(text)

            results = matcher.run(
                frames=[frame],
                # TODO: convert to Pydantic model
                template_frames=template_frames,
                template_boxes=template_bboxes,
                template_labels=template_labels,
                template_texts=template_texts,
                metadata=ocr_results,
                score_threshold=0.80,
                max_overlap=0.5,
                max_objects=2,
                window_size=window_size,
                downscale_factor=1,
            )
            print(results)
            rows = []
            bboxes = []
            labels = []
            scores = []

            cv_raw_image = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)

            for result in results:
                snippet = frame[
                    result.bbox[1] : result.bbox[1] + result.bbox[3],
                    result.bbox[0] : result.bbox[0] + result.bbox[2],
                ]
                snippet = Image.fromarray(snippet)
                snippet.save(f"/tmp/template/snippet_{i}.png")

                rows.append(
                    {
                        "snippet": image_formatter(snippet),
                        "label": result.label,
                        "score": result.score,
                        "x": result.bbox[0],
                        "y": result.bbox[1],
                        "w": result.bbox[2],
                        "h": result.bbox[3],
                    }
                )
                bboxes.append(result.bbox)
                labels.append(result.label)
                scores.append(result.score)

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                column_config=column_configuration,
            )

            BaseTemplateMatcher.visualize_object_predictions(
                bboxes, labels, scores, cv_raw_image, 0
            )
            st.image(cv_raw_image)


if __name__ == "__main__":
    main()

# streamlit run app.py --server.fileWatcherType none
# https://discuss.streamlit.io/t/streamlit-components-security-and-a-five-month-quest-to-ship-a-single-line-of-code/9019
