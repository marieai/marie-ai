from typing import List, Union

import numpy as np
from pydantic import BaseModel


class DocumentBoundaryPrediction(BaseModel):
    label: str
    detected: bool
    mode: str
    aligned_image: Union[np.ndarray, None]
    boundary_bbox: List[int]
    score: float
    visualization_image: Union[np.ndarray, None]  # Added for visualization purposes

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self):
        return {
            "label": self.label,
            "detected": self.detected,
            "mode": self.mode,
            "aligned_image": (
                self.aligned_image.tolist() if self.aligned_image is not None else None
            ),
            "visualization_image": (
                self.visualization_image.tolist()
                if self.visualization_image is not None
                else None
            ),
            "boundary_bbox": self.boundary_bbox,
            "score": self.score,
        }
