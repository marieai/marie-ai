from typing import List, Optional

from marie.extract.models.definition import WorkUnit
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.readers import MetaReader
from marie.extract.structures.unstructured_document import UnstructuredDocument


def create_execution_context(
    work_unit: WorkUnit, page_numbers: Optional[List[int]] = None
) -> ExecutionContext:
    """
    Creates an execution context for processing a work unit, potentially using specific
    page numbers for targeted operations.

    The function constructs an `UnstructuredDocument` using the provided frames and
    metadata from the `work_unit`. If `page_numbers` are supplied, only the frames and
    metadata corresponding to those pages will be used. The resulting execution context
    includes the document ID, template, and the constructed document.

    :param work_unit: The work unit containing the required frames, metadata, and template for processing.
    :param page_numbers: A list of specific page indices to filter the frames and metadata, or None to process all pages.
    :return: An execution context containing the document ID, template, and the unstructured document representation.
    """
    frames = work_unit.frames
    metadata = work_unit.metadata
    template = work_unit.template

    if page_numbers:
        frames = [frame for idx, frame in enumerate(frames) if idx in page_numbers]
        metadata = [meta for idx, meta in enumerate(metadata) if idx in page_numbers]

    doc: UnstructuredDocument = MetaReader.from_data(frames=frames, ocr_meta=metadata)
    return ExecutionContext(doc_id=work_unit.doc_id, template=template, document=doc)
