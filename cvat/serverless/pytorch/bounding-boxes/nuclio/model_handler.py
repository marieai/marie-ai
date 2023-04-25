import torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_image = None
        self.predictor = BoxProcessorUlimDit(
            models_dir="/etc/marie/model_zoo/unilm/dit/text_detection",
            cuda=torch.cuda.is_available(),
        )

    def handle(self, image):
        self.latest_image = image

        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = self.predictor.extract_bounding_boxes("cvat", "field", image, PSMode.SPARSE)

        return boxes
