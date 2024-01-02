from typing import List


def scale_bounding_box(
    box: List[int], width_scale: float = 1.0, height_scale: float = 1.0
) -> List[int]:
    """
    Scale a bounding box by a given factor.

    Example usage (the bounding box coordinates are normalized to the range of 0-1000):

    .. code-block:: python

        width, height = image.shape[1], image.shape[0]
        width_scale = 1000 / width
        height_scale = 1000 / height
        boxes_normalized = [scale_bounding_box(box, width_scale, height_scale) for box in boxes]

    :param box: the bounding box to scale
    :param width_scale: the factor to scale the width by
    :param height_scale: the factor to scale the height by
    :return: the scaled bounding box
    """
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]
