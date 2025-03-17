from PIL import Image
from streamlit_drawable_canvas import st_canvas

from marie.utils.resize_image import resize_image_progressive


class ImageUtils:
    def __init__(self):
        pass

    def read_image(self, uploaded_image, progressive_rescale=False, scale=75):
        """Read the uploaded image"""
        raw_image = Image.open(uploaded_image).convert("RGB")
        print("progressive_rescale", progressive_rescale, scale)
        if progressive_rescale:
            raw_image = resize_image_progressive(
                raw_image,
                reduction_percent=scale / 100,
                reductions=2,
                return_intermediate_states=False,
            )
        width, height = raw_image.size
        return raw_image, (width, height)

    def resize_image_for_canvas(self, raw_image, square=960):
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
        print("raw_size", raw_size)
        print("resized_size", resized_size)
        print("resized_boxes", resized_boxes)

        raw_width, raw_height = raw_size
        resized_width, resized_height = resized_size
        raw_boxes = []

        for box in resized_boxes:
            left, top, right, bottom = box
            raw_left = int(left * raw_width / resized_width)
            raw_top = int(top * raw_height / resized_height)
            raw_right = int(right * raw_width / resized_width)
            raw_bottom = int(bottom * raw_height / resized_height)
            box2 = [raw_left, raw_top, raw_right, raw_bottom]
            raw_boxes.append(box2)
        return raw_boxes
