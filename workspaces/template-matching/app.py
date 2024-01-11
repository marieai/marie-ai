import logging

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

    def resize_image(self, raw_image, square=544):
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
            fill_color="rgba(255,0, 0, 0.1)",
            stroke_width=2,
            stroke_color="rgba(255,0,0,1)",
            background_color="rgba(0,0,0,1)",
            background_image=resized_image,
            update_streamlit=update_streamlit,
            height=height,
            width=width,
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
    utils = ImageUtils()

    st.set_page_config(layout="wide")
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s%(levelname)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    st.title("Template matching")

    col1_opt, col2_opt, col3_opt, col4_opt = st.columns(4)
    with st.container():
        with col1_opt:
            max_matches_number = st.number_input(
                'Max matches', min_value=1, max_value=10, value=1, step=1, format='%d'
            )
            score_threshold_number = st.number_input(
                'Match threshold',
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                format='%d',
            )

            col1_x, col2_x = st.columns(2)
            with col1_x:
                window_size_h = st.number_input(
                    'Window size height(px)',
                    min_value=128,
                    max_value=512,
                    value=256,
                    step=1,
                    format='%d',
                )
            with col2_x:
                window_size_w = st.number_input(
                    'Window size width(px)',
                    min_value=128,
                    max_value=512,
                    value=256,
                    step=1,
                    format='%d',
                )

        with col2_opt:
            st.write('Max number is ', max_matches_number)
            st.write('Score threashold', score_threshold_number)
        with col3_opt:
            uploaded_image = st.file_uploader(
                "Upload template source: ",
                type=["jpg", "jpeg", "png", "webp"],
                key="source",
            )
            uploaded_target = st.file_uploader(
                "Upload document to match: ",
                type=["jpg", "jpeg", "png", "webp"],
                key="target",
            )
        with col4_opt:
            submit = st.button("Submit")

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            if uploaded_image is not None:
                raw_image, raw_size = utils.read_image(uploaded_image)
                resized_image, resized_size = utils.resize_image(raw_image)
                # read bbox input
                canvas_result = utils.get_canvas(
                    resized_image, key="canvas-source", update_streamlit=True
                )
        with col2:
            if uploaded_target is not None:
                raw_image_target, raw_size_target = utils.read_image(uploaded_target)
                resized_image_target, resized_size_target = utils.resize_image(
                    raw_image_target
                )
                canvas_resultZ = utils.get_canvas(
                    resized_image_target, key="canvas-target", update_streamlit=False
                )

    with st.container():
        if submit:
            resized_boxes = utils.get_resized_boxes(canvas_result)
            # left_upper point and right_lower point : [x1, y1, x2, y2]
            raw_boxes = utils.get_raw_boxes(resized_boxes, raw_size, resized_size)
            # convert bbxo to [x1, y1, w, h]
            raw_boxes_xywh = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in raw_boxes
            ]

            sample = {
                "image": raw_image,
                "bboxes": raw_boxes,
                "boxes_xywh": raw_boxes_xywh,
            }

            st.write("Received the following sample:")
            st.success(sample)


if __name__ == "__main__":
    main()

# streamlit run app.py --server.fileWatcherType none
# https://discuss.streamlit.io/t/streamlit-components-security-and-a-five-month-quest-to-ship-a-single-line-of-code/9019
