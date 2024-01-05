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
