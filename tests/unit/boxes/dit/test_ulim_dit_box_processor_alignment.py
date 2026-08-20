from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import torch

from marie.boxes.dit.ulim_dit_box_processor import BoxProcessorUlimDit


class _FakePredBoxes:
    def __init__(self, boxes: list[list[float]]) -> None:
        self.tensor = torch.tensor(boxes, dtype=torch.float32)

    def to(self, _device: torch.device) -> '_FakePredBoxes':
        return self

    def __len__(self) -> int:
        return len(self.tensor)


class _FakeInstances:
    def __init__(
            self,
            boxes: list[list[float]],
            classes: list[int],
            scores: list[float],
    ) -> None:
        self.pred_boxes = _FakePredBoxes(boxes)
        self.pred_classes = torch.tensor(classes, dtype=torch.int64)
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def has(self, field: str) -> bool:
        return hasattr(self, field)


def test_psm_sparse_step_keeps_bbox_class_score_alignment() -> None:
    processor = BoxProcessorUlimDit.__new__(BoxProcessorUlimDit)
    processor.logger = Mock()
    processor.cpu_device = torch.device('cpu')

    instances = _FakeInstances(
        boxes=[
            [10, 10, 40, 20],
            [10, 10, 40, 20],
            [60, 10, 90, 20],
        ],
        classes=[0, 1, 0],
        scores=[0.9, 0.8, 0.7],
    )
    processor.predictor = lambda _img: {'instances': instances}

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes, classes, scores = processor.psm_sparse_step(image, 0, 0)

    assert len(bboxes) == 3
    assert len(classes) == 3
    assert len(scores) == 3
    np.testing.assert_allclose(
        bboxes,
        np.array(
            [
                [10.0, 10.0, 40.0, 20.0],
                [10.0, 10.0, 40.0, 20.0],
                [60.0, 10.0, 90.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(classes, np.array([0, 1, 0], dtype=np.int64))
    np.testing.assert_allclose(scores, np.array([0.9, 0.8, 0.7], dtype=np.float32))

    triples = [
        (tuple(np.round(box, 4).tolist()), int(cls), round(float(score), 4))
        for box, cls, score in zip(bboxes, classes, scores)
    ]
    assert triples == [
        ((10.0, 10.0, 40.0, 20.0), 0, 0.9),
        ((10.0, 10.0, 40.0, 20.0), 1, 0.8),
        ((60.0, 10.0, 90.0, 20.0), 0, 0.7),
    ]


def test_extract_bounding_boxes_keeps_line_numbers_sorted_with_fragments(monkeypatch: Any) -> None:
    processor = BoxProcessorUlimDit.__new__(BoxProcessorUlimDit)
    processor.logger = Mock()
    processor.work_dir = '/tmp/boxes'
    processor.strict_box_segmentation = False

    monkeypatch.setattr(
        'marie.boxes.dit.ulim_dit_box_processor.create_dirs',
        lambda *_args, **_kwargs: ('/tmp', '/tmp', '/tmp', '/tmp'),
    )

    def _fake_find_line_number(_lines: list[list[int]], box: list[int]) -> int:
        return 2 if box[1] < 20 else 1

    find_line_number_spy = Mock(side_effect=_fake_find_line_number)
    monkeypatch.setattr(
        'marie.boxes.dit.ulim_dit_box_processor.find_line_number',
        find_line_number_spy,
    )

    def _fake_psm_sparse(*_args: Any, **_kwargs: Any):
        bboxes = np.array(
            [
                [100, 10, 120, 20],
                [100, 10, 120, 20],
                [10, 30, 30, 40],
            ],
            dtype=np.float32,
        )
        # Higher score is non-text duplicate for the same box.
        scores = np.array([0.8, 0.95, 0.9], dtype=np.float32)
        classes = np.array([0, 1, 0], dtype=np.int64)
        return bboxes, classes, scores, [[0, 0, 150, 50]], classes

    processor.psm_sparse = _fake_psm_sparse

    image = np.ones((64, 160, 3), dtype=np.uint8) * 255
    rects, fragments, line_numbers, prediction, _line_bboxes = (
        processor.extract_bounding_boxes('qid', 'kid', image)
    )

    assert len(fragments) == 2
    assert list(line_numbers) == [1, 2]
    assert rects[0].tolist() == [10, 30, 20, 10]
    assert rects[1].tolist() == [100, 10, 20, 10]
    assert len(prediction['bboxes']) == 3
    assert find_line_number_spy.call_count == 2
    assert [call.args[1] for call in find_line_number_spy.call_args_list] == [
        [100, 10, 20, 10],
        [10, 30, 20, 10],
    ]


def test_extract_bounding_boxes_collapses_rounded_duplicates(monkeypatch: Any) -> None:
    processor = BoxProcessorUlimDit.__new__(BoxProcessorUlimDit)
    processor.logger = Mock()
    processor.work_dir = '/tmp/boxes'
    processor.strict_box_segmentation = False

    monkeypatch.setattr(
        'marie.boxes.dit.ulim_dit_box_processor.create_dirs',
        lambda *_args, **_kwargs: ('/tmp', '/tmp', '/tmp', '/tmp'),
    )
    monkeypatch.setattr(
        'marie.boxes.dit.ulim_dit_box_processor.find_line_number',
        lambda _lines, _box: 1,
    )

    def _fake_psm_sparse(*_args: Any, **_kwargs: Any):
        bboxes = np.array(
            [
                [100.00011, 10.0, 120.00011, 20.0],
                [100.00014, 10.0, 120.00014, 20.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.8, 0.9], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int64)
        return bboxes, classes, scores, [[0, 0, 150, 50]], classes

    processor.psm_sparse = _fake_psm_sparse

    image = np.ones((64, 160, 3), dtype=np.uint8) * 255
    _rects, fragments, _line_numbers, prediction, _line_bboxes = (
        processor.extract_bounding_boxes('qid', 'kid', image)
    )

    assert len(fragments) == 1
    assert len(prediction['bboxes']) == 1
    np.testing.assert_allclose(prediction['scores'], np.array([0.9], dtype=np.float32))


def test_extract_bounding_boxes_keeps_boxes_across_rounding_boundary(monkeypatch: Any) -> None:
    processor = BoxProcessorUlimDit.__new__(BoxProcessorUlimDit)
    processor.logger = Mock()
    processor.work_dir = '/tmp/boxes'
    processor.strict_box_segmentation = False

    monkeypatch.setattr(
        'marie.boxes.dit.ulim_dit_box_processor.create_dirs',
        lambda *_args, **_kwargs: ('/tmp', '/tmp', '/tmp', '/tmp'),
    )
    monkeypatch.setattr(
        'marie.boxes.dit.ulim_dit_box_processor.find_line_number',
        lambda _lines, _box: 1,
    )

    def _fake_psm_sparse(*_args: Any, **_kwargs: Any):
        bboxes = np.array(
            [
                [100.00014, 10.0, 120.00014, 20.0],
                [100.00016, 10.0, 120.00016, 20.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.8, 0.9], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int64)
        return bboxes, classes, scores, [[0, 0, 150, 50]], classes

    processor.psm_sparse = _fake_psm_sparse

    image = np.ones((64, 160, 3), dtype=np.uint8) * 255
    _rects, fragments, _line_numbers, prediction, _line_bboxes = (
        processor.extract_bounding_boxes('qid', 'kid', image)
    )

    assert len(fragments) == 2
    assert len(prediction['bboxes']) == 2
