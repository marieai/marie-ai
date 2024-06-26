from typing import Union

import cv2
import numpy as np
import PIL
from PIL import Image


def resize_image(
    image, desired_size, color=(255, 255, 255), keep_max_size=False
) -> tuple:
    """Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    """

    # check if the image is already of the desired size
    if image.shape[0] == desired_size[0] and image.shape[1] == desired_size[1]:
        return image, (0, 0, image.shape[1], image.shape[0])

    size = image.shape[:2]
    if keep_max_size:
        # size = (max(size[0], desired_size[0]), max(size[1], desired_size[1]))
        h = size[0]
        w = size[1]
        dh = desired_size[0]
        dw = desired_size[1]

        if w > dw and h < dh:
            delta_h = max(0, desired_size[0] - size[0])
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left = 40
            right = 40
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
            )
            size = image.shape[:2]
            # cv2.imwrite("/tmp/marie/box_framed_keep_max_size.png", image)
            return image, (left, top, size[1], size[0])

    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(
            image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC
        )
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    # convert top, bottom, left, right to x, y, w, h
    x, y, w, h = left, top, size[1], size[0]

    # cv2.imwrite("/tmp/dim/box_framed.png", image)
    return image, (x, y, w, h)


def resize_image_progressive(
    image: Union[np.ndarray, PIL.Image.Image],
    reduction_percent: float,
    reductions: int = 2,
    return_intermediate_states=False,
) -> Union[np.ndarray, PIL.Image.Image]:
    """Resize an image progressively by reducing the size of the image"""
    if image is None:
        raise Exception("Input image can't be empty")

    return_format = "np"
    if isinstance(image, PIL.Image.Image):
        return_format = "PIL"
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_size = image.shape[:2]
    desired_size = [dim - int(dim * reduction_percent) for dim in original_size]

    reduction_w, reduction_h = [
        (max_dim - min_dim) // reductions
        for max_dim, min_dim in zip(original_size, desired_size)
    ]

    reduced_image = image.copy()
    intermediate_states = []

    for _ in range(reductions):
        reduced_size = [
            curr_dim - reduction
            for curr_dim, reduction in zip(
                reduced_image.shape[:2], (reduction_w, reduction_h)
            )
        ]
        reduced_image = cv2.resize(
            reduced_image,
            (reduced_size[1], reduced_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        intermediate_states.append(reduced_image)
    if return_format == "PIL":
        reduced_image = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2RGB)
        reduced_image = Image.fromarray(reduced_image)

    if return_intermediate_states:
        return reduced_image, intermediate_states

    return reduced_image
