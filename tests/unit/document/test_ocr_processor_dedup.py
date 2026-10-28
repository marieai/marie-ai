from __future__ import annotations

from typing import Any

import numpy as np

from marie.document.ocr_processor import OcrProcessor


class _StubOcrProcessor(OcrProcessor):
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        super().__init__(work_dir='/tmp/icr', cuda=False)
        self._outputs = outputs

    def is_available(self) -> bool:
        return True

    def recognize_from_fragments(self, image_fragments):
        return self._outputs


def test_recognize_returns_list_lines_contract() -> None:
    processor = _StubOcrProcessor(
        outputs=[
            {'text': 'ALPHA', 'confidence': 0.9},
            {'text': 'ALPHA', 'confidence': 0.8},
            {'text': 'BETA', 'confidence': 0.95},
        ]
    )

    image = np.ones((40, 80, 3), dtype=np.uint8) * 255
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 10, 10]]
    fragments = [
        image[0:10, 0:10].copy(),
        image[0:10, 0:10].copy(),
        image[0:10, 20:30].copy(),
    ]
    lines = [1, 1, 1]

    result, _overlay = processor.recognize(
        'qid', 'kid', image, boxes, fragments, lines, return_overlay=False
    )

    assert len(result['words']) == 3
    assert [word['text'] for word in result['words']] == ['ALPHA', 'ALPHA', 'BETA']
    assert isinstance(result['lines'], list)
    assert all(isinstance(line, dict) for line in result['lines'])
    assert all(isinstance(line['text'], str) for line in result['lines'])
    assert result['lines'][0]['text'] == 'ALPHA ALPHA BETA'
