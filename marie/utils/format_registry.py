"""Canonical names used for document format detection.

This module describes recognizable inputs, not extraction support. Runtime
support comes from the document extraction plugin capability snapshot and the
OCR input loader.
"""

# Raster image formats (fed directly to the vision pipeline)
IMAGE_FORMATS = {
    "png",
    "jpeg",
    "tiff",
    "bmp",
    "gif",
    "webp",
    "heif",
}

PDF_FORMAT = "pdf"
TIFF_FORMAT = "tiff"

# Extension (without dot) -> canonical format name
EXT_TO_FORMAT: dict[str, str] = {
    # Images
    "png": "png",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "tif": "tiff",
    "tiff": "tiff",
    "bmp": "bmp",
    "gif": "gif",
    "webp": "webp",
    "heif": "heif",
    "heic": "heif",
    # PDF
    "pdf": "pdf",
    # Semantic documents
    "docx": "docx",
    "xlsx": "xlsx",
    "pptx": "pptx",
    "html": "html",
    "htm": "html",
    "md": "markdown",
    "markdown": "markdown",
    "epub": "epub",
    "msg": "msg",
    "rst": "rst",
    "csv": "csv",
    "tsv": "tsv",
    "eml": "eml",
    "xml": "xml",
    # Detectable legacy documents (runtime support is capability-driven)
    "doc": "doc",
    "xls": "xls",
    "ppt": "ppt",
    "odt": "odt",
    "ods": "ods",
    "odp": "odp",
    "rtf": "rtf",
    # Source formats
    "tex": "latex",
    "latex": "latex",
    "djvu": "djvu",
}

# MIME type -> canonical format name
MIME_TO_FORMAT: dict[str, str] = {
    # Images
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/heif": "heif",
    "image/heic": "heif",
    # PDF
    "application/pdf": "pdf",
    # Documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/html": "html",
    "text/markdown": "markdown",
    "application/epub+zip": "epub",
    "application/vnd.ms-outlook": "msg",
    "text/x-rst": "rst",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "message/rfc822": "eml",
    "application/xml": "xml",
    "text/xml": "xml",
    # Detectable legacy documents (runtime support is capability-driven)
    "application/msword": "doc",
    "application/vnd.ms-excel": "xls",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/rtf": "rtf",
    # Source formats
    "application/x-latex": "latex",
    "image/vnd.djvu": "djvu",
}

ALL_DETECTABLE_FORMATS = frozenset(
    set(EXT_TO_FORMAT.values()) | set(MIME_TO_FORMAT.values())
)
