import numpy as np
import pytest

from marie.utils.image_utils import ensure_max_page_size


def test_max_page_base():
    frame = np.random.randint(0, 255, (3200, 2550), dtype=np.uint8)
    src_frames = [frame]

    changed, frames = ensure_max_page_size(src_frames, expand_ratio=0)

    assert changed is False
    assert frames[0].shape == (3200, 2550)


def test_max_page_no_expansion():
    frame = np.random.randint(0, 255, (3200, 2600), dtype=np.uint8)
    src_frames = [frame]

    changed, frames = ensure_max_page_size(src_frames, expand_ratio=0)

    assert changed
    assert frames[0].shape == (3138, 2550)


def test_max_page_with_expansion():
    frame = np.random.randint(0, 255, (3200, 2600), dtype=np.uint8)
    src_frames = [frame]

    changed, frames = ensure_max_page_size(src_frames)

    assert changed is False
    assert frames[0].shape == (3200, 2600)


def test_max_page_001():
    frame = np.random.randint(0, 255, (4171, 2569), dtype=np.uint8)
    src_frames = [frame]

    changed, frames = ensure_max_page_size(src_frames)

    assert changed is True
    assert frames[0].shape == (3200, 2600)


def test_downsample_001():
    # read image using cv2
    import cv2
    cv2_img = cv2.imread('~/template-matching/TID-101220/TO-COMPARE/199026220-0001.png')
    # resize image proportionally
    resized = cv2.resize(cv2_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)

    # save the image
    cv2.imwrite(
        '~/template-matching/TID-101220/TO-COMPARE/199026220-0001-TEST-50-LANCHOS.png',
        resized)

    # now resize using scale pyramids
    # read image using cv2
    scale_1 = cv2.resize(cv2_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LANCZOS4)
    scale_2 = cv2.resize(scale_1, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LANCZOS4)
    scale_3 = cv2.resize(scale_2, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LANCZOS4)

    # save the image
    cv2.imwrite('~/template-matching/TID-101220/TO-COMPARE/199026220-0001-TEST-75-S1.png',
                scale_1)
    cv2.imwrite('~/template-matching/TID-101220/TO-COMPARE/199026220-0001-TEST-75-S2.png',
                scale_2)
    cv2.imwrite('~/template-matching/TID-101220/TO-COMPARE/199026220-0001-TEST-75-S3.png',
                scale_3)
