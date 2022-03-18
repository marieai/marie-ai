from boxes.box_processor import BoxProcessor, PSMode


class BoxProcessorTextFuseNet(BoxProcessor):
    """ "TextFuseNet box processor responsible for extracting bounding boxes for given documents"""

    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = "./models/textfusenet",
        cuda: bool = False,
    ):
        super().__init__(work_dir, models_dir, cuda)
        print("Box processor [textfusenet, cuda={}]".format(cuda))

    def extract_bounding_boxes(self, _id, key, img, psm=PSMode.SPARSE):
        pass
