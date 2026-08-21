from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite
from tifftools import read_tiff

from marie.utils.tiff_ops import (
    merge_tiff_frames_with_splits_ifd,
    merge_tiff_paths_ifd,
)


def _source_tiffs(source_dir: Path, count: int = 4) -> None:
    source_dir.mkdir()
    for index in range(count):
        imwrite(
            source_dir / f'page_{index:05}.tif', np.full((4, 4), index, dtype=np.uint8)
        )


def test_merge_tiff_paths_preserves_ifds(tmp_path: Path) -> None:
    source_dir = tmp_path / 'source'
    _source_tiffs(source_dir, 2)
    output = tmp_path / 'merged.tif'

    merge_tiff_paths_ifd(
        [str(path) for path in sorted(source_dir.glob('*.tif'))], output
    )

    assert len(read_tiff(str(output))['ifds']) == 2


def test_merge_tiff_frames_with_splits(tmp_path: Path) -> None:
    source_dir = tmp_path / 'source'
    output_dir = tmp_path / 'output'
    _source_tiffs(source_dir)

    outputs = merge_tiff_frames_with_splits_ifd(
        source_dir,
        [2],
        output_dir,
        sort_key=lambda path: int(Path(path).stem.rsplit('_', 1)[-1]),
    )

    assert len(outputs) == 2
    assert len(read_tiff(str(output_dir / outputs[0]))['ifds']) == 2
    assert len(read_tiff(str(output_dir / outputs[1]))['ifds']) == 2


@pytest.mark.parametrize('split_indices', [[0], [2, 2], [-1], [4]])
def test_merge_tiff_frames_rejects_invalid_boundaries(
    tmp_path: Path, split_indices: list[int]
) -> None:
    source_dir = tmp_path / 'source'
    _source_tiffs(source_dir)

    with pytest.raises(ValueError):
        merge_tiff_frames_with_splits_ifd(
            source_dir, split_indices, tmp_path / 'output', sort_key=lambda path: path
        )


def test_merge_tiff_frames_rejects_empty_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        merge_tiff_frames_with_splits_ifd(
            tmp_path / 'source', [1], tmp_path / 'output', sort_key=lambda path: path
        )
