from boxes.box_processor import BoxProcessor
from document.icr_processor import IcrProcessor
from utils.utils import ensure_exists

def init()->None:
    print('Initializng processors on load')

    global box_processor
    global icr_processor

    box_processor = BoxProcessor(work_dir=ensure_exists('/tmp/boxes'), models_dir='./models/craft')
    icr_processor = IcrProcessor(work_dir=ensure_exists('/tmp/icr'), cuda=False)

init()
