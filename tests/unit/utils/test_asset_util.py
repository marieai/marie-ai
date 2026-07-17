import json
import logging
from pathlib import Path

import numpy as np
import pytest

from marie.utils import asset_util


def test_prepare_asset_directory_refreshes_metadata_when_frames_are_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'assets'
    frames_dir = root / 'frames'
    frames_dir.mkdir(parents=True)
    (frames_dir / '00001.png').write_bytes(b'cached')
    metadata_path = root / 'document.meta.json'
    metadata_path.write_text('{"ocr": ["stale"]}')
    source = tmp_path / 'document.pdf'
    source.write_bytes(b'%PDF')

    monkeypatch.setattr(
        asset_util, 'create_working_dir', lambda *args, **kwargs: str(root)
    )

    def download_asset(**_kwargs: object) -> str:
        metadata_path.write_text('{"ocr": ["fresh"]}')
        return str(metadata_path)

    monkeypatch.setattr(asset_util, 'download_asset', download_asset)

    _, _, returned_metadata = asset_util.prepare_asset_directory.__wrapped__(
        frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        local_path=str(source),
        ref_id='document',
        ref_type='longextract-bench',
        logger=logging.getLogger(__name__),
    )

    assert returned_metadata == str(metadata_path)
    assert json.loads(metadata_path.read_text())['ocr'] == ['fresh']
