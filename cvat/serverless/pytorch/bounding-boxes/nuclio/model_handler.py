import numpy as np
import cv2
import torch


def convert_mask_to_polygon(mask):
    contours = None

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon


class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = "model.pth"
        self.model_type = "vit_h"
        self.latest_image = None
        self.predictor = None

    def handle(self, image, pos_points, neg_points):
        object_mask = None

        polygon = convert_mask_to_polygon(object_mask)
        return object_mask, polygon
