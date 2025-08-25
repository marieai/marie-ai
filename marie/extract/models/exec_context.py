from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from marie.extract.models.base import Page
from marie.extract.models.definition import Template
from marie.extract.structures import UnstructuredDocument


class ExecutionContext(BaseModel):
    template: Optional[Template] = None
    document: Optional[UnstructuredDocument] = None
    pages: List[Page] = []  # Page
    tree: Optional[Any] = None
    doc_id: str
    metadata: Optional[Dict] = None
    output_dir: Path

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return (
            f"ExecutionContext(doc_id={self.doc_id}, "
            f"template={self.template}, "
            f"document={self.document}, "
            f"metadata_keys={list(self.metadata.keys()) if self.metadata else []})"
            f"output_dir={self.output_dir}"
        )

    def get_template(self) -> Template:
        return self.template
