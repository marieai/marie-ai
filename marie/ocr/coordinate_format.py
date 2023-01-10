from enum import Enum

import numpy as np


class CoordinateFormat(Enum):
    """Output format for the words
    defaults to : xywh
    """

    XYWH = "xywh"  # Default
    XYXY = "xyxy"

    @staticmethod
    def from_value(value: str):
        if value is None:
            return CoordinateFormat.XYWH
        for data in CoordinateFormat:
            if data.value == value.lower():
                return data
        return CoordinateFormat.XYWH

    @staticmethod
    def convert(
        box: np.ndarray, from_mode: "CoordinateFormat", to_mode: "CoordinateFormat"
    ) -> np.ndarray:
        """
        Args:
            box: can be a 4-tuple,
            from_mode, to_mode (CoordinateFormat)

        Ref : Detectron boxes
        Returns:
            The converted box of the same type.
        """
        arr = np.array(box)
        assert arr.shape == (4,), "CoordinateFormat.convert takes either a 4-tuple/list"

        if from_mode == to_mode:
            return box

        original_type = type(box)
        original_shape = arr.shape
        arr = arr.reshape(-1, 4)

        if to_mode == CoordinateFormat.XYXY and from_mode == CoordinateFormat.XYWH:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == CoordinateFormat.XYXY and to_mode == CoordinateFormat.XYWH:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise RuntimeError("Cannot be here!")

        return original_type(arr.flatten())
