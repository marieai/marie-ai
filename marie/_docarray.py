try:
    from docarray import BaseDoc  # Direct export for type hints
    from docarray import BaseDoc as Document
    from docarray import DocList  # Direct export for type hints like DocList[SomeType]
    from docarray import DocList as DocumentArray
    from docarray.documents.legacy import LegacyDocument

    docarray_v2 = True

except ImportError:
    from docarray import Document, DocumentArray

    BaseDoc = None  # Not available in docarray v1
    DocList = None  # Not available in docarray v1
    LegacyDocument = None  # Not available in docarray v1
    docarray_v2 = False
