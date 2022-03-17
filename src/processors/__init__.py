from boxes.box_processor_craft import BoxProcessorCraft
from utils.utils import ensure_exists
from document.icr_processor import IcrProcessor


def init() -> None:
    print("Initializing processors on load")

    global box_processor
    global icr_processor

    box_processor = BoxProcessorCraft(work_dir=ensure_exists("/tmp/boxes"), models_dir="./models/craft")
    icr_processor = IcrProcessor(work_dir=ensure_exists("/tmp/icr"), cuda=False)


init()
