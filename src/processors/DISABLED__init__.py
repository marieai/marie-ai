from boxes.craft_box_processor import BoxProcessorCraft
from utils.utils import ensure_exists
from document.craft_icr_processor import CraftIcrProcessor


def init() -> None:
    print("Initializing processors on load")

    global box_processor
    global icr_processor

    box_processor = BoxProcessorCraft(work_dir=ensure_exists("/tmp/boxes"), models_dir="./model_zoo/craft")
    icr_processor = CraftIcrProcessor(work_dir=ensure_exists("/tmp/icr"), cuda=False)


init()
