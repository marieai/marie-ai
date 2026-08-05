import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import filetype
import numpy as np
from docarray import DocList
from PIL import Image

from marie.api.docs import DOC_KEY_PAGE_NUMBER, MarieDoc
from marie.common.file_io import StrOrBytesPath
from marie.logging_core.predefined import default_logger as logger
from marie.storage import StorageManager
from marie.utils.format_registry import (
    ALL_DETECTABLE_FORMATS,
    EXT_TO_FORMAT,
    MIME_TO_FORMAT,
)
from marie.utils.utils import ensure_exists

TYPES_TO_EXT = {
    "png": "png",
    "jpeg": "jpg",
    "tiff": "tif",
    "bmp": "bmp",
    "gif": "gif",
    "webp": "webp",
    "heif": "heif",
    "pdf": "pdf",
    "docx": "docx",
    "xlsx": "xlsx",
    "pptx": "pptx",
    "html": "html",
    "markdown": "md",
    "epub": "epub",
    "msg": "msg",
    "rst": "rst",
    "csv": "csv",
    "doc": "doc",
    "xls": "xls",
    "ppt": "ppt",
    "odt": "odt",
    "ods": "ods",
    "odp": "odp",
    "rtf": "rtf",
    "latex": "tex",
    "djvu": "djvu",
}

OCR_RASTER_FORMATS = frozenset(
    {'pdf', 'png', 'jpeg', 'tiff', 'bmp', 'gif', 'webp', 'heif'}
)
_OCR_RASTER_EXTENSIONS = frozenset(
    extension
    for extension, canonical in EXT_TO_FORMAT.items()
    if canonical in OCR_RASTER_FORMATS and canonical != 'pdf'
)

_DEFAULT_MAX_RASTER_PAGES = 500
_DEFAULT_MAX_RASTER_DECODED_BYTES = 8 * 1024**3


class UnsupportedOcrInputError(ValueError):
    """Raised when a source cannot be rasterized by Marie OCR."""


class DocumentTooLargeError(ValueError):
    """Raised when decoding a document would exceed configured limits."""

    retryable = False


def get_document_type(file_path: str) -> str:
    """Detect document format using magic bytes, then extension fallback."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Tier 1: magic-byte detection
    guess = filetype.guess(file_path)
    if guess is not None:
        mime = guess.mime
        if mime in MIME_TO_FORMAT:
            return MIME_TO_FORMAT[mime]

    # Tier 2: extension fallback
    _, ext = os.path.splitext(file_path)
    ext = ext.lstrip(".").lower()
    if ext in EXT_TO_FORMAT:
        return EXT_TO_FORMAT[ext]

    raise ValueError(
        f"Unrecognized file type for '{file_path}'. Detectable: {ALL_DETECTABLE_FORMATS}"
    )


def convert_frames(frames, img_format):
    """each frame needs to be converted to RGB format"""
    converted = []
    for frame in frames:
        # cv to pil
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 2:
                conv = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                conv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if img_format == "pil":
                conv = Image.fromarray(frame.copy())
            converted.append(conv)
        else:
            converted.append(frame.copy())
    return converted


def supports_ocr_input(
    source: StrOrBytesPath, *, format_hint: str | None = None
) -> bool:
    """Return whether the source can be rasterized for Marie OCR."""
    if format_hint:
        normalized = format_hint.lower().lstrip('.')
        canonical = EXT_TO_FORMAT.get(normalized, normalized)
        return canonical in OCR_RASTER_FORMATS

    path = Path(os.path.expanduser(os.fspath(source)))
    if path.is_dir():
        return any(_numeric_image_files(path))
    try:
        canonical = get_document_type(str(path))
    except (FileNotFoundError, ValueError):
        return False
    return canonical in OCR_RASTER_FORMATS


def load_document(
    file_path: str,
    img_format: str = "cv",
    pages: Sequence[int] | None = None,
    *,
    dpi: int = 200,
) -> dict[str, Any]:
    """Load an OCR-compatible document into ordered raster frames."""
    loaded_frames = _load_document_frames(file_path, pages, dpi=dpi)
    if img_format == 'pil':
        frames: list[Any] = [Image.fromarray(frame.copy()) for frame in loaded_frames]
    elif img_format == 'cv':
        frames = loaded_frames
    else:
        raise ValueError(f'Unsupported image format: {img_format!r}')
    return {'mode': 'frames', 'frames': frames}


def load_image(img_path, img_format: str = "cv") -> (bool, List[np.ndarray]):
    """Backward-compatible wrapper. Returns (bool, List[np.ndarray])."""
    if img_path is None:
        return False, None

    result = load_document(img_path, img_format)
    frames = result['frames']
    return (True, frames) if frames else (False, [])


def frames_from_docs(
    docs: DocList[MarieDoc], field: Optional[str] = None
) -> List[np.ndarray]:
    """Convert DocList[MarieDoc] to Numpy Array"""
    if docs is None:
        raise ValueError("No documents provided to convert to array")
    frames = []
    if field is None:
        field = 'tensor'

    for index, doc in enumerate(docs):
        frame = getattr(doc, field)
        if frame is None:
            raise ValueError(
                f'Document at index {index} has no {field!r}; '
                'semantic results cannot be used as OCR frames'
            )
        frames.append(frame)

    # each tensor can be of different size that is why we are using 'concatenate' instead of 'vstack'
    # concat = np.concatenate(frames, axis=None)
    assert len(frames) == len(docs)
    return frames


def docs_from_file(
    path: StrOrBytesPath, pages: Optional[List[int]] = None
) -> DocList[MarieDoc]:
    """
    Create DocumentArray from document file. This will create one document per page.

    :param path:  path to document file
    :param pages:  list of pages to extract from document NONE or empty list will extract all pages from document
    :return: DocumentArray with tensor content
    """
    if path is not None:
        path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found : {path}")

    result = load_document(path, pages=pages)
    return _docs_from_frames(result['frames'], pages)


def fetch_asset_to_temp(asset_key: str) -> Tuple[str, str]:
    """Download a remote asset to /tmp/marie preserving its extension.

    Returns (local_path, file_type) without parsing/rendering the document.

    :param asset_key: asset key / URI to the resource
    :return: tuple of (local file path, detected document format)
    """
    uri = asset_key

    if not StorageManager.can_handle(uri, allow_native=True):
        raise Exception(
            f"Unable to read file from {uri} no suitable storage manager configured"
        )

    # Ensure the directory exists
    ensure_exists(f"/tmp/marie")

    # Read remote file to a byte array. Keep the source extension so
    # get_document_type's extension fallback works for formats without
    # magic bytes (csv, markdown, html, ...).
    suffix = os.path.splitext(uri)[1]
    with tempfile.NamedTemporaryFile(
        dir="/tmp/marie", suffix=suffix, delete=False
    ) as temp_file_out:
        print(f"Reading file from {uri} to {temp_file_out.name}")

        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
        if not connected:
            logger.error(f"Error restoring assets : Could not connect to S3")
            raise ValueError("Error restoring assets : Could not connect to S3")

        if not StorageManager.exists(uri):
            raise ValueError(f"Remote file does not exist : {uri}")

        StorageManager.read_to_file(uri, temp_file_out, overwrite=True)
        path = temp_file_out.name

    # Detect type only after the with-block closes the handle: read_to_file
    # writes through the still-open buffered file object, so reading `path`
    # inside the block sees a partially-flushed (for small files: empty) file.
    file_type = get_document_type(path)

    return path, file_type


def docs_from_asset(
    asset_key: str, pages: Optional[List[int]] = None, return_file_path: bool = False
) -> Union[DocList[MarieDoc], Tuple[DocList[MarieDoc], str]]:
    """
    Create DocumentArray from image file. This will create one document per page in the image
    file, if the image is large and has many pages this can be very memory intensive.

    :param asset_key: asset key to the resource
    :param pages: list of pages to extract from the document. NONE or empty list will extract all pages from document
    :param return_file_path: whether to return the path of the downloaded file
    :return: DocList[MarieDoc] with tensor content or a tuple (DocList[MarieDoc], file_path) if return_file_path is True
    """

    path, _file_type = fetch_asset_to_temp(asset_key)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found : {path}")

    result = load_document(path, pages=pages)
    docs = _docs_from_frames(result['frames'], pages)

    if return_file_path:
        return docs, path
    return docs


def frames_from_file(img_path: StrOrBytesPath) -> List[np.ndarray]:
    """Create Numpy frame array from image file or directory of image files."""
    result = load_document(img_path)
    return result['frames']


def is_array_like(obj: Any) -> bool:
    """Check if object is array like"""
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return True
    return False


def docs_from_image(src: Union[Any, List]) -> DocList[MarieDoc]:
    """Create DocumentArray from image or array like object.
    Numpy ndarray or PIl Image ar supported
    """
    frames = src
    if not is_array_like(src):
        frames = [src]

    docs = DocList[MarieDoc]()
    for i, frame in enumerate(frames):
        if isinstance(frame, Image.Image):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        doc = MarieDoc(tensor=frame)
        doc.tags[DOC_KEY_PAGE_NUMBER] = i
        docs.append(doc)

    return docs


def _docs_from_frames(
    frames: Sequence[np.ndarray], pages: Sequence[int] | None
) -> DocList[MarieDoc]:
    selected = _normalize_pages(pages)
    page_numbers = selected if selected is not None else list(range(len(frames)))
    docs = DocList[MarieDoc]()
    for page_number, frame in zip(page_numbers, frames):
        doc = MarieDoc(tensor=frame)
        doc.tags[DOC_KEY_PAGE_NUMBER] = page_number
        docs.append(doc)
    return docs


def _load_document_frames(
    source: StrOrBytesPath,
    pages: Sequence[int] | None,
    *,
    dpi: int,
) -> list[np.ndarray]:
    path = Path(os.path.expanduser(os.fspath(source)))
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    if path.is_dir():
        frames = _load_directory_frames(path)
        return _select_frames(frames, pages)

    canonical = get_document_type(str(path))
    if canonical not in OCR_RASTER_FORMATS:
        raise UnsupportedOcrInputError(
            f'OCR document loading does not support {canonical or path.suffix or path}'
        )
    if canonical == 'pdf':
        return _load_pdf_frames(path, pages, dpi=dpi)

    return _load_raster_image_frames(path, pages)


def _load_pdf_frames(
    path: Path, pages: Sequence[int] | None, *, dpi: int
) -> list[np.ndarray]:
    from pdf2image import convert_from_path

    selected = _normalize_pages(pages)
    if selected is None:
        images = convert_from_path(str(path), dpi=dpi)
    else:
        images = []
        invalid_pages = []
        for page in selected:
            rendered = convert_from_path(
                str(path),
                dpi=dpi,
                first_page=page + 1,
                last_page=page + 1,
            )
            if not rendered:
                invalid_pages.append(page)
                continue
                # raise IndexError(f'PDF page index out of range: {page}')
            images.append(rendered[0])
        if invalid_pages:
            logger.warning(f'PDF page indices out of range: {invalid_pages}')
    return [_rgb_array(image) for image in images]


def _load_raster_image_frames(
    path: Path,
    pages: Sequence[int] | None = None,
) -> list[np.ndarray]:
    _register_heif()
    with Image.open(path) as image:
        try:
            frame_count = image.n_frames
        except AttributeError:
            frame_count = 1
        selected = _normalize_pages(pages)
        if selected is not None and any(page >= frame_count for page in selected):
            logger.warning(
                f'Raster page indices out of range: {selected}, '
                f'image has {frame_count} frames'
            )
            selected = [page for page in selected if page < frame_count]

        indices: Sequence[int] = range(frame_count) if selected is None else selected

        for page in indices:
            if page >= frame_count:
                raise IndexError(f'Page index out of range: {page}')

        max_pages = int(
            os.environ.get(
                'MARIE_MAX_RASTER_PAGES',
                str(_DEFAULT_MAX_RASTER_PAGES),
            )
        )
        if max_pages > 0 and len(indices) > max_pages:
            raise DocumentTooLargeError(
                f'Raster document has {len(indices)} selected pages; '
                f'limit is {max_pages}'
            )

        max_decoded_bytes = int(
            os.environ.get(
                'MARIE_MAX_RASTER_DECODED_BYTES',
                str(_DEFAULT_MAX_RASTER_DECODED_BYTES),
            )
        )
        if max_decoded_bytes > 0:
            estimated_decoded_bytes = 0
            for index in indices:
                image.seek(index)
                width, height = image.size
                estimated_decoded_bytes += width * height * 3

            if estimated_decoded_bytes > max_decoded_bytes:
                raise DocumentTooLargeError(
                    'Raster document requires approximately '
                    f'{estimated_decoded_bytes} decoded RGB bytes; '
                    f'limit is {max_decoded_bytes}'
                )

        frames: list[np.ndarray] = []
        for index in indices:
            image.seek(index)
            frames.append(_rgb_array(image))
        return frames


def _load_directory_frames(path: Path) -> list[np.ndarray]:
    files = _numeric_image_files(path)
    if not files:
        raise UnsupportedOcrInputError(
            f'No numerically named OCR images found in directory: {path}'
        )
    frames = []
    for file_path in files:
        frames.extend(_load_raster_image_frames(file_path))
    return frames


def _numeric_image_files(path: Path) -> list[Path]:
    candidates = []
    for file_path in path.iterdir():
        if not file_path.is_file() or not file_path.stem.isdigit():
            continue
        if file_path.suffix.lower().lstrip('.') not in _OCR_RASTER_EXTENSIONS:
            continue
        candidates.append(file_path)
    return sorted(candidates, key=lambda item: int(item.stem))


def _select_frames(
    frames: list[np.ndarray], pages: Sequence[int] | None
) -> list[np.ndarray]:
    selected = _normalize_pages(pages)
    if selected is None:
        return frames
    result = []
    invalid_pages = []
    for page in selected:
        if page >= len(frames):
            invalid_pages.append(page)
            continue
            # raise IndexError(f'Page index out of range: {page}')
        result.append(frames[page])
    if invalid_pages:
        logger.warning(f'Page indices out of range: {invalid_pages}')
    return result


def _normalize_pages(pages: Sequence[int] | None) -> list[int] | None:
    if pages is None or len(pages) == 0:
        return None
    normalized = []
    for page in pages:
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError(f'Page indices must be non-negative integers: {page!r}')
        normalized.append(page)
    return normalized


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()
