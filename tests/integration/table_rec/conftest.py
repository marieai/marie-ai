import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
from PIL import Image, ImageDraw

from marie.components.table_rec import TableRecPredictor
from marie.constants import __model_path__


@pytest.fixture(scope="session")
def table_rec_predictor() -> TableRecPredictor:
    model_path = os.path.join(__model_path__, "table_recognition", "2025_02_18")
    table_rec_predictor = TableRecPredictor(checkpoint=model_path)
    yield table_rec_predictor
    del table_rec_predictor


@pytest.fixture()
def test_image():
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello World", fill="black", font_size=72)
    draw.text((10, 200), "This is a sentence of text.\nNow it is a paragraph.\nA three-line one.", fill="black",
              font_size=24)
    return image


@pytest.fixture()
def test_image_tall():
    image = Image.new("RGB", (4096, 4096), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello World", fill="black", font_size=72)
    draw.text((4000, 4000), "This is a sentence of text.\n\nNow it is a paragraph.\n\nA three-line one.", fill="black",
              font_size=24)
    return image
