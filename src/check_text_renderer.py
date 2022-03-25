import os

import numpy as np
import tqdm

import cv2
from renderer.text_renderer import TextRenderer
from boxes.box_processor import PSMode
from utils.utils import ensure_exists


if True:
    from boxes.craft_box_processor import BoxProcessorCraft
    from boxes.textfusenet_box_processor import BoxProcessorTextFuseNet
    from document.craft_icr_processor import CraftIcrProcessor


def cal_mean_std(images_dir):
    """
    :param images_dir:
    :return:
    """
    img_filenames = os.listdir(images_dir)
    m_list, s_list = [], []
    for img_filename in img_filenames:
        img = cv2.imread(images_dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)

        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
        print(m_list)
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print('mean: ', m[0][::-1])
    print('std:  ', s[0][::-1])
    return m


if __name__ == '__main__':

    work_dir_boxes = ensure_exists('/tmp/boxes')
    work_dir_icr = ensure_exists('/tmp/icr')
    ensure_exists('/tmp/fragments')

    img_path = './assets/psm/word/0001.png'
    img_path = './assets/english/Scanned_documents/Picture_029.tif'
    # img_path = './assets/english/Scanned_documents/t2.tif'
    img_path = './assets/english/Scanned_documents/Picture_010.tif'
    img_path = './assets/english/Lines/002.png'
    # img_path = './assets/english/Lines/001.png'
    # img_path = './assets/english/Lines/003.png'
    # img_path = './assets/english/Lines/005.png'
    # img_path = './assets/english/Lines/004.png'
    img_path = './assets/private/PID_576_7188_0_149495857_page_0002.tif'

    # cal_mean_std('./assets/english/Scanned_documents/')

    if not os.path.exists(img_path):
        raise Exception(f'File not found : {img_path}')

    if True:
        key = img_path.split('/')[-1]
        image = cv2.imread(img_path)
        mean, std = cv2.meanStdDev(image)

        print(mean)
        print(std)

        box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir='./models/craft', cuda=False)
        # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./models/fusenet', cuda=False)
        icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=False)

        boxes, img_fragments, lines, _ = box.extract_bounding_boxes(
            key, 'field', image, PSMode.LINE)

        result, overlay_image = icr.recognize(key, 'test', image, boxes, img_fragments, lines)

        print("Testing text render")

        renderer = TextRenderer(config={"preserve_interword_spaces": True})
        renderer.render(image, result)
