import torch


def get_center_crop_coords(width: int, height: int, crop_width: int, crop_height: int):
    x1 = (width - crop_width) // 2 if width > crop_width else 0
    x2 = x1 + crop_width
    y1 = (height - crop_height) // 2 if height > crop_height else 0
    y2 = y1 + crop_height
    return x1, y1, x2, y2


def convert_box_to_integral(box_filter: torch.Tensor):
    haar_multiplier = torch.tensor([[1, -1], [-1, 1]]).float()
    integral_filter = torch.zeros(box_filter.shape[0] + 1, box_filter.shape[1] + 1)
    for i in range(box_filter.shape[0]):
        for j in range(box_filter.shape[1]):
            integral_filter[i : i + 2, j : j + 2] += box_filter[i, j] * haar_multiplier
    return integral_filter
