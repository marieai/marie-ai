from boxes.box_processor import BoxProcessor, PSMode


class BoxProcessorTextFuseNet(BoxProcessor):
    """ "TextFuseNet box processor responsible for extracting bounding boxes for given documents"""

    def __init__(self):
        super().__init__()

    def extract_bounding_boxes(self, _id, key, img, psm=PSMode.SPARSE):
        pass
