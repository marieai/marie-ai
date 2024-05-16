import base64
import difflib
import io
import json
import logging
import os
from io import BytesIO

import cv2
import pandas as pd
import streamlit as st
import streamlit_shortcuts
from canvas_util import ImageUtils
from PIL import Image
from streamlit import session_state as ss
from streamlit_drawable_canvas import st_canvas

from marie.utils.json import load_json_file, store_json_object

src_dir = os.path.expanduser("~/tmp/ocr-diffs/v5/ocr1")
output_dir = os.path.expanduser("~/tmp/ocr-diffs/json_data")

os.makedirs(output_dir, exist_ok=True)

json_files = [
    os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".json")
]
json_files.sort()

json_files = [
    os.path.join(src_dir, f)
    for f in os.listdir(src_dir)
    if f.endswith(".json")
    and not os.path.exists(os.path.join(output_dir, os.path.splitext(f)[0] + ".txt"))
]
json_files.sort()


def load_json(file_index):
    # Load the JSON file
    with open(json_files[file_index]) as f:
        data = json.load(f)
    return data


def display_image(base64_string, scale=1):
    decoded_image = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(decoded_image))
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    st.image(image)


def save_image_and_text(image_data, text, base_name):
    # Save the image and text to output_dir
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image.save(os.path.join(output_dir, f"{base_name}.png"))
    with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
        f.write(text)


def accept_word(snippet, word, base_name):
    def callback():
        # st.write(f"You accepted {word}")
        st.session_state.selected_word = word
        save_image_and_text(snippet, word, base_name)
        st.session_state.current_file_index += 1

    return callback


def get_canvas(self, resized_image, key="canvas", update_streamlit=True, mode="rect"):
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


def create_initial_drawing(bboxes):
    template = {
        "type": "rect",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": 0,
        "top": 0,
        "width": 0,
        "height": 0,
        "fill": "rgba(255, 165, 0, 0.3)",
        "stroke": "rgba(255,0,0,1)",
        "strokeWidth": 2,
        "strokeLineCap": "butt",
        "strokeDashOffset": 0,
        "strokeLineJoin": "miter",
        "strokeMiterLimit": 4,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "opacity": 1,
        "backgroundColor": "",
        "fillRule": "nonzero",
        "paintFirst": "fill",
        "globalCompositeOperation": "source-over",
        "skewX": 0,
        "skewY": 0,
        "rx": 0,
        "ry": 0,
    }
    objects = []
    for bbox in bboxes:
        obj = template.copy()
        obj["left"] = bbox[0]
        obj["top"] = bbox[1]
        obj["width"] = bbox[2]
        obj["height"] = bbox[3]
        objects.append(obj)

    return {
        "version": "4.4.0",
        "objects": objects,
    }


def get_diff_boxes(diffs):
    bboxes = []
    for diff in diffs:
        word1 = diff["word1"]  # xywh
        box = word1["box"]
        bboxes.append(box)

    print(bboxes)
    return bboxes


def main():
    st.set_page_config(
        page_title="OCR-DIFF",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "current_file_index" not in st.session_state:
        st.session_state.current_file_index = 0
    if "selected_word" not in st.session_state:
        st.session_state.selected_word = ""

    uploaded_image = st.sidebar.file_uploader(
        "Upload template source: ",
        type=["jpg", "jpeg", "png", "webp", "tiff", "tif"],
        key="source",
    )

    utils = ImageUtils()

    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s%(levelname)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    st.sidebar.write("Matching parameters")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    submit = st.sidebar.button("Submit")

    # Convert bounding boxes to the format required by st_canvas
    # initial_drawing = load_json_file(os.path.join(output_dir, "canvas.json"))
    word_diffs = load_json_file(os.path.join(output_dir, "diff.json"))

    raw_boxes = get_diff_boxes(word_diffs)
    initial_drawing = create_initial_drawing(raw_boxes)

    if "selected_row_index" not in ss:
        ss.selected_row_index = None

    # for bbox in bboxes
    canvas_result = None
    col1, col2 = st.columns([2, 1])
    with st.container(border=True):
        with col1:
            # st.header("Source image")
            if uploaded_image is not None:
                raw_image, raw_size = utils.read_image(uploaded_image, False, 0.75)
                resized_image, resized_size = utils.resize_image_for_canvas(raw_image)
                # read bbox input
                canvas_result = utils.get_canvas(
                    resized_image,
                    key="canvas-source",
                    update_streamlit=True,
                    mode="rect",
                    initial_drawing=initial_drawing,
                )

        with col2:
            st.write("Diff-Boxes")

            def image_to_base64(img) -> str:
                with BytesIO() as buffer:
                    img.save(buffer, "png")  # or 'jpeg'
                    return base64.b64encode(buffer.getvalue()).decode()

            def image_formatter(img) -> str:
                return f"data:image/png;base64,{image_to_base64(img)}"

            # max_width = max([box[2] - box[0] for box in raw_boxes])
            max_width = max([box[2] for box in raw_boxes])
            column_configuration = {
                "Select": st.column_config.CheckboxColumn(),
                "snippet": st.column_config.ImageColumn(
                    "snippet", help="Document snippet", width=max_width
                ),
            }
            rows = []
            for word_diff in word_diffs:
                snippet = word_diff["snippet"]
                txt_1 = word_diff["word1"]["text"]
                txt_2 = word_diff["word2"]["text"]
                conf_1 = word_diff["confidence1"]
                conf_2 = word_diff["confidence2"]

                rows.append(
                    {
                        "snippet": f"data:image/png;base64,{snippet}",
                        "engine-1": txt_1,
                        "engine-2": txt_2,
                        "conf-1": conf_1,
                        "conf-2": conf_2,
                    }
                )

            df = pd.DataFrame(rows)
            # Add a new column for radio buttons
            # df["Select"] = [False] * len(df)

            # def add_click_events(df):
            #     df_temp = df.copy()
            #     df_temp["Select"] = False  # Temporary column
            #     edited_df = st.data_editor(
            #         df_temp,
            #         column_config=column_configuration,
            #         disabled=df.columns,
            #     )
            #     selected_rows = edited_df[edited_df["Select"]].drop("Select", axis=1)
            #     return selected_rows
            #
            # selection = add_click_events(df)
            # st.write("Selected rows:", selection)

            # https://github.com/streamlit/streamlit/pull/8411

            st.dataframe(
                df,
                use_container_width=True,
                column_config=column_configuration,
                # on_select="rerun",
                # selection_mode="single-row",
            )

            selected_row = df.iloc[df.index]
            st.text_input("Selected row:", str(selected_row))

    if submit:
        st.write("Submitted")
        if canvas_result is not None:
            st.write(canvas_result.json_data)
            store_json_object(
                canvas_result.json_data, os.path.join(output_dir, "canvas.json")
            )


if __name__ == "__main__":
    main()
