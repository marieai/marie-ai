from __future__ import annotations

from typing import TYPE_CHECKING

from marie.extract.annotators.context_provider import ContextProvider, ProcessingUnit
from marie.extract.registry import register_context_provider

from .page_parsing import (
    ANNOTATOR_NAME,
    PageBinding,
    parse_run_parameters,
    prompt_variables,
    validate_document_bindings,
)

if TYPE_CHECKING:
    from marie.kernel.context import RunContext

    from marie.extract.structures.unstructured_document import UnstructuredDocument


@register_context_provider(
    name='parsebench',
    target_annotators=[ANNOTATOR_NAME],
)
class ParseBenchPageContextProvider(ContextProvider):
    def __init__(
        self,
        run_context: RunContext | None,
        annotator_name: str,
        mode: str = 'per-page',
    ) -> None:
        super().__init__(run_context, annotator_name, mode='per-page')
        self.parameters = parse_run_parameters(run_context)

    def get_eligible_pages(self, document: UnstructuredDocument) -> set[int]:
        validate_document_bindings(document, self.parameters)
        return {page.local_page_number for page in self.parameters.pages}

    def get_processing_units(
        self, document: UnstructuredDocument
    ) -> list[ProcessingUnit]:
        validate_document_bindings(document, self.parameters)
        return [
            ProcessingUnit(
                page_number=page.local_page_number,
                index=None,
                data=page.model_dump(mode='json'),
            )
            for page in self.parameters.pages
        ]

    def get_variables(
        self,
        document: UnstructuredDocument,
        page_number: int,
        unit: ProcessingUnit | None = None,
    ) -> dict[str, str]:
        if unit is None or unit.page_number != page_number:
            raise ValueError('ParseBench requires the actual page processing unit')
        binding = PageBinding.model_validate(unit.data)
        expected = self.parameters.pages[page_number - 1]
        if binding != expected:
            raise ValueError('processing unit page binding does not match RunContext')
        return prompt_variables(document, binding)
