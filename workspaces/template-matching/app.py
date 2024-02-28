import logging

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


class ImageUtils:
    def __init__(self):
        pass

    def read_image(self, uploaded_image):
        """Read the uploaded image"""
        raw_image = Image.open(uploaded_image)
        width, height = raw_image.size
        return raw_image, (width, height)

    def resize_image(self, raw_image, square=960):
        """Resize the mask so it fits inside a 544x544 square"""
        width, height = raw_image.size

        # Check if the image resolution is larger than
        if width > square or height > square:
            # Calculate the aspect ratio
            aspect_ratio = width / height

            # Calculate the new dimensions to fit within 960x544 frame
            if aspect_ratio > square / square:
                resized_width = square
                resized_height = int(resized_width / aspect_ratio)
            else:
                resized_height = square
                resized_width = int(resized_height * aspect_ratio)

            # Resize the image
            resized_image = raw_image.resize((resized_width, resized_height))
            return resized_image, (resized_width, resized_height)
        else:
            return raw_image, (width, height)

    def get_canvas(self, resized_image, key="canvas", update_streamlit=True):
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
            drawing_mode="rect",
            key=key,
        )
        return canvas_result

    def get_resized_boxes(self, canvas_result):
        """Get the resized boxes from the canvas result"""
        objects = canvas_result.json_data["objects"]
        resized_boxes = []
        for object in objects:
            left, top = object["left"], object["top"]  # upper left corner
            width, height = object["width"], object["height"]  # box width and height
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


def main():
    # st.set_page_config(page_title="", layout="wide")

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
            
            div[class^='block-container'] { padding-top: 0rem; }
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

    # stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    # drawing_mode = "rect"
    # stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    # bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    # # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    max_matches_number = st.sidebar.number_input(
        "Max matches", min_value=1, max_value=10, value=1, step=1, format="%d"
    )
    score_threshold_number = st.sidebar.number_input(
        "Match threshold",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        format="%d",
    )

    st.sidebar.divider()

    window_size_h = st.sidebar.number_input(
        "Window size height(px)",
        min_value=128,
        max_value=512,
        value=256,
        step=1,
        format="%d",
    )
    window_size_w = st.sidebar.number_input(
        "Window size width(px)",
        min_value=128,
        max_value=512,
        value=256,
        step=1,
        format="%d",
    )
    st.sidebar.divider()

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
                    resized_image, key="canvas-source", update_streamlit=True
                )
        with col2:
            # st.header("Matching target")

            if uploaded_target is not None:
                raw_image_target, raw_size_target = utils.read_image(uploaded_target)
                resized_image_target, resized_size_target = utils.resize_image(
                    raw_image_target
                )
                canvas_resultZ = utils.get_canvas(
                    resized_image_target, key="canvas-target", update_streamlit=False
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
            resized_boxes = utils.get_resized_boxes(canvas_result)
            # left_upper point and right_lower point : [x1, y1, x2, y2]
            raw_boxes = utils.get_raw_boxes(resized_boxes, raw_size, resized_size)
            # convert bbox to [x1, y1, w, h]
            raw_boxes_xywh = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in raw_boxes
            ]

            matching_request = {
                "image": raw_image,
                "bboxes": raw_boxes,
                "boxes_xywh": raw_boxes_xywh,
            }

            # create a pandas dataframe
            # Boolean to resize the dataframe, stored as a session state variable
            st.checkbox("Use container width", value=False, key="use_container_width")

            st.write("Boxes - original/converted:")
            rows = []
            for s, r, z in zip(resized_boxes, raw_boxes, raw_boxes_xywh):
                rows.append(
                    {
                        "sx1": s[0],
                        "sy1": s[1],
                        "sx2": s[2],
                        "sy2": s[3],
                        "x1": r[0],
                        "y1": r[1],
                        "x2": r[2],
                        "y2": r[3],
                        "x": z[0],
                        "y": z[1],
                        "w": z[2],
                        "h": z[3],
                    }
                )

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=st.session_state.use_container_width)

            st.write("Received the following sample:")
            st.success(matching_request)


if __name__ == "__main__":
    main()

# streamlit run app.py --server.fileWatcherType none
# https://discuss.streamlit.io/t/streamlit-components-security-and-a-five-month-quest-to-ship-a-single-line-of-code/9019
