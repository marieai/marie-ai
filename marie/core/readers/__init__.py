"""
Data Connectors for LlamaIndex.

This module contains the data connectors for LlamaIndex. Each connector inherits
from a `BaseReader` class, connects to a data source, and loads Document objects
from that data source.

You may also choose to construct Document objects manually, for instance
in our `Insert How-To Guide <../how_to/insert.html>`_. See below for the API
definition of a Document - the bare minimum is a `text` property.

"""

from marie.core.readers.base import ReaderConfig
from marie.core.readers.download import download_loader

# readers
from marie.core.readers.file.base import (
    SimpleDirectoryReader,
    FileSystemReaderMixin,
)
from marie.core.readers.string_iterable import StringIterableReader
from marie.core.schema import Document

__all__ = [
    "SimpleDirectoryReader",
    "FileSystemReaderMixin",
    "ReaderConfig",
    "Document",
    "StringIterableReader",
    "download_loader",
]
