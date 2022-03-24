import os

import cv2
from boxes.box_processor_craft import BoxProcessorCraft, PSMode
from document.icr_processor_craft import IcrProcessorCraft
from utils.utils import ensure_exists


if __name__ == '__main__':

    work_dir_boxes = ensure_exists('/tmp/boxes')
    work_dir_icr = ensure_exists('/tmp/icr')
    img_path = './assets/psm/word/0001.png'

    if not os.path.exists(img_path):
        raise Exception(f'File not found : {img_path}')

    key = img_path.split('/')[-1]
    snippet = cv2.imread(img_path)

    box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir='./models/craft')
    icr = IcrProcessorCraft(work_dir=work_dir_icr, cuda=False)

    boxes, img_fragments, lines, _ = box.extract_bounding_boxes(
        key, 'field', snippet, PSMode.WORD)

    print(boxes)
    icr.recognize(key, 'test', snippet, boxes, img_fragments, lines)
