import numpy as np
import pytest

from marie.utils.patches import patchify


def test_patchify_2d_non_overlapping_grid() -> None:
    image = np.arange(16).reshape(4, 4)

    patches = patchify(image, (2, 2), step=2)

    assert patches.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(patches[0, 0], image[0:2, 0:2])
    np.testing.assert_array_equal(patches[0, 1], image[0:2, 2:4])
    np.testing.assert_array_equal(patches[1, 0], image[2:4, 0:2])
    np.testing.assert_array_equal(patches[1, 1], image[2:4, 2:4])


def test_patchify_3d_full_channel_patch_roundtrip() -> None:
    image = np.arange(512 * 512 * 3, dtype=np.uint32).reshape(512, 512, 3)

    patches = patchify(image, (128, 128, 3), step=128)
    reconstructed = patches[:, :, 0].transpose(0, 2, 1, 3, 4).reshape(image.shape)

    assert patches.shape == (4, 4, 1, 128, 128, 3)
    np.testing.assert_array_equal(reconstructed, image)


def test_patchify_rejects_patch_larger_than_image() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="must fit inside array shape"):
        patchify(image, (16, 16), step=8)
