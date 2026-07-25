import io
import zipfile
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from marie.api import _detect_type_from_bytes, store_temp_file


def _zip_bytes(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("word/document.xml", "docx"),
        ("xl/workbook.xml", "xlsx"),
        ("ppt/presentation.xml", "pptx"),
    ],
)
def test_detects_openxml_zip_subtypes(member, expected):
    assert _detect_type_from_bytes(_zip_bytes({member: "content"})) == expected


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("application/epub+zip", "epub"),
        ("application/vnd.oasis.opendocument.text", "odt"),
        ("application/vnd.oasis.opendocument.spreadsheet", "ods"),
        ("application/vnd.oasis.opendocument.presentation", "odp"),
    ],
)
def test_detects_mimetype_zip_subtypes(mime, expected):
    assert _detect_type_from_bytes(_zip_bytes({"mimetype": mime})) == expected


def test_explicit_format_hint_admits_text_without_guessing_content():
    assert _detect_type_from_bytes(b"heading,amount\nA,10", format_hint="csv") == "csv"


def test_rejects_unknown_format_hint():
    with pytest.raises(ValueError, match="Unrecognized format hint"):
        _detect_type_from_bytes(b"data", format_hint="unknown")


def test_rejects_oversized_zip_mimetype_member():
    data = _zip_bytes({"mimetype": "x" * 257})

    with pytest.raises(ValueError, match="Could not detect"):
        _detect_type_from_bytes(data)


def test_store_temp_file_preserves_encoded_image_bytes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    encoded_jpeg = b"\xff\xd8\xff\xe0encoded-jpeg\xff\xd9"
    monkeypatch.setattr("marie.api.ensure_exists", lambda _: str(tmp_path))

    temp_file, _ = store_temp_file(
        encoded_jpeg,
        queue_id="test-queue",
        file_type="jpeg",
        store_raw=False,
    )

    assert temp_file.endswith(".jpg")
    assert Path(temp_file).read_bytes() == encoded_jpeg
