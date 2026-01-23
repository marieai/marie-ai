from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type

if TYPE_CHECKING:
    from marie_kernel.context import RunContext

    from marie.extract.structures.unstructured_document import UnstructuredDocument


@dataclass
class ProcessingUnit:
    """
    Represents a single unit of work for context-aware processing.

    A processing unit can represent either:
    - A whole page (index is None) - for legacy/default behavior
    - A specific unit on a page (index is set) - for per-unit processing

    Attributes:
        page_number: 1-indexed page number
        index: Index within page's units (0-based), None for whole page
        data: The specific unit dict when processing per-unit
    """

    page_number: int
    index: Optional[int] = None
    data: Optional[Dict[str, Any]] = None

    @property
    def output_suffix(self) -> str:
        """
        Generate suffix for output filename.

        Returns:
            "_t{index}" when index is set, empty string for whole page.
        """
        if self.index is not None:
            return f"_t{self.index}"
        return ""


@dataclass
class ContextProviderInfo:
    """Metadata about a registered context provider."""

    name: str
    cls: Type["ContextProvider"]
    target_annotators: List[str] = field(default_factory=list)


class ContextProvider(ABC):
    """
    Protocol for context injection into prompt templates.

    Providers subscribe to annotators via the decorator:
    @register_context_provider(
        name="table_claims",
        target_annotators=["claim-extract"]
    )

    Implementations are responsible for:
    - Querying upstream task data via RunContext
    - Determining which pages are eligible for processing
    - Formatting and injecting context into prompts

    Subclasses must implement:
    - get_eligible_pages(): which pages to process
    - get_variables(): variable substitutions for injection
    """

    def __init__(
        self,
        run_context: Optional["RunContext"],
        annotator_name: str,
    ):
        """
        Initialize the context provider.

        Args:
            run_context: RunContext for accessing upstream task results.
            annotator_name: Name of the annotator this provider is injecting into.
        """
        self.run_context = run_context
        self.annotator_name = annotator_name

    @abstractmethod
    def get_eligible_pages(self, document: "UnstructuredDocument") -> Set[int]:
        """
        Return set of 1-indexed page numbers that this provider can process.

        Pages not in this set will be skipped during annotation.

        Args:
            document: The document being processed, provides access to
                     page_count, metadata, and other document properties.

        Returns:
            Set of 1-indexed page numbers eligible for processing.
        """
        pass

    def get_processing_units(
        self, document: "UnstructuredDocument"
    ) -> List["ProcessingUnit"]:
        """
        Return list of processing units for this provider.

        Default implementation returns one unit per eligible page (backward compatible).
        Override for per-unit processing.

        Args:
            document: The document being processed.

        Returns:
            List of ProcessingUnit instances representing work items.
        """
        return [
            ProcessingUnit(page_number=p)
            for p in sorted(self.get_eligible_pages(document))
        ]

    @abstractmethod
    def get_variables(
        self,
        document: "UnstructuredDocument",
        page_number: int,
        unit: Optional["ProcessingUnit"] = None,
    ) -> Dict[str, str]:
        """
        Get all variable substitutions for a specific page or processing unit.

        Args:
            document: The document being processed.
            page_number: 1-indexed page number.
            unit: Optional ProcessingUnit for per-unit context. If provided,
                  use unit.data for single-unit processing.

        Returns:
            Dict mapping placeholder names to their replacement values.
            Empty string values will replace the placeholder with nothing.

        Example:
            return {
                "TABLE_CONTEXT_CLAIMS": json.dumps(tables),
                "TABLE_COUNT": str(len(tables)),
                "HAS_TABLES": "true" if tables else "false",
            }
        """
        pass

    def inject(
        self,
        prompt: str,
        document: "UnstructuredDocument",
        page_number: int,
        unit: Optional["ProcessingUnit"] = None,
    ) -> str:
        """
        Inject context into prompt template.

        Uses get_variables() to get all substitutions, then performs
        string replacement for each variable.

        Args:
            prompt: The prompt template containing placeholder variables.
            document: The document being processed.
            page_number: 1-indexed page number.
            unit: Optional ProcessingUnit for per-unit context.

        Returns:
            Prompt with all placeholders replaced.
        """
        variables = self.get_variables(document, page_number, unit)
        for var_name, value in variables.items():
            prompt = prompt.replace(var_name, value)
        return prompt


class ContextProviderManager:
    """
    Manages context providers for an annotator.

    Automatically discovers providers that subscribe to the annotator name.
    """

    def __init__(
        self,
        run_context: Optional["RunContext"],
        annotator_name: str,
    ):
        """
        Initialize the context provider manager.

        Args:
            run_context: RunContext for accessing upstream task results.
            annotator_name: Name of the annotator - providers that target this
                          annotator will be auto-discovered.
        """
        from marie.extract.registry import component_registry
        from marie.logging_core.predefined import default_logger as logger

        self.providers: List[ContextProvider] = []
        self._logger = logger
        # Auto-discover providers that target this annotator
        provider_infos = component_registry.get_providers_for_annotator(annotator_name)

        for info in provider_infos:
            try:
                provider = info.cls(run_context, annotator_name)
                self.providers.append(provider)
                logger.info(
                    f"Activated context provider '{info.name}' for annotator '{annotator_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize context provider '{info.name}': {e}"
                )

    def get_eligible_pages(self, document: "UnstructuredDocument") -> Set[int]:
        """
        Return intersection of all providers' eligible pages.

        If no providers are configured, returns all pages.

        Args:
            document: The document being processed.

        Returns:
            Set of 1-indexed page numbers eligible for processing.
        """
        if not self.providers:
            return set(range(1, document.page_count + 1))

        eligible: Optional[Set[int]] = None
        for provider in self.providers:
            pages = provider.get_eligible_pages(document)
            if eligible is None:
                eligible = pages
            else:
                eligible = eligible.intersection(pages)

        return eligible or set()

    def get_processing_units(
        self, document: "UnstructuredDocument"
    ) -> List[ProcessingUnit]:
        """
        Get processing units from the first provider.

        If no providers are configured, returns one unit per page.

        Args:
            document: The document being processed.

        Returns:
            List of ProcessingUnit instances representing work items.
        """
        if not self.providers:
            # No providers: one unit per page
            return [
                ProcessingUnit(page_number=p) for p in range(1, document.page_count + 1)
            ]

        return self.providers[0].get_processing_units(document)

    def inject_all(
        self,
        prompt: str,
        document: "UnstructuredDocument",
        page_number: int,
        unit: Optional[ProcessingUnit] = None,
    ) -> str:
        """
        Inject context from all providers into prompt.

        Args:
            prompt: The prompt template.
            document: The document being processed.
            page_number: 1-indexed page number.
            unit: Optional ProcessingUnit for per-unit context.

        Returns:
            Prompt with all context placeholders replaced.
        """
        for provider in self.providers:
            prompt = provider.inject(prompt, document, page_number, unit)
        return prompt

    def inject_for_unit(
        self,
        prompt: str,
        document: "UnstructuredDocument",
        unit: ProcessingUnit,
    ) -> str:
        """
        Inject context for a specific processing unit.

        Convenience method that extracts page_number from unit.

        Args:
            prompt: The prompt template.
            document: The document being processed.
            unit: ProcessingUnit specifying what to process.

        Returns:
            Prompt with all context placeholders replaced.
        """
        return self.inject_all(prompt, document, unit.page_number, unit)

    def has_providers(self) -> bool:
        """Check if any providers are configured."""
        return len(self.providers) > 0
