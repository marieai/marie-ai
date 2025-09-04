from collections import deque
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from marie.excepts import BadConfigSource, ProcessingError
from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection, MatchSectionType
from marie.extract.models.span import Span
from marie.extract.registry import component_registry
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.structured_region import StructuredRegion
from marie.logging_core.logger import MarieLogger


class MatchSectionRegionProcessorVisitor(BaseProcessingVisitor):
    """
    A visitor that uses configurable processors for parsing different types of regions.
    Each processor can specialize in handling specific region types (remarks, codes, notes, etc.).
    """

    def __init__(self, enabled: bool, fail_fast: bool = True):
        super().__init__(enabled)
        self.logger = MarieLogger(context=self.__class__.__name__)
        self._role_to_processor = {}
        self.fail_fast = fail_fast

    def visit(self, context: ExecutionContext, parent: MatchSection) -> None:
        self.logger.info("----------------------------------------")
        self.logger.info("Processing MatchSectionRegionParserVisitor")
        queue = deque([parent])
        while queue:
            current = queue.popleft()
            if current is None:
                continue
            self.logger.info(f'---- Extracting from : {current.type}')
            if current.type == MatchSectionType.CONTENT:
                self.process_section(context, parent, current)
            queue.extend(current.sections)
        self.logger.info("Finished processing MatchSectionRegionParserVisitor")
        self.logger.info("----------------------------------------")

    def process_section(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        """
        Processes a given section within a document layer to extract field values
        based on defined selectors and annotations.

        Args:
            context (ExecutionContext): The execution context containing the document.
            parent (MatchSection): The parent section in the document hierarchy.
            section (MatchSection): The current section to process.
        """
        assert context is not None, "Execution context must not be None."
        assert section is not None, "Section must not be None."
        assert parent is not None, "Parent section must not be None."
        assert (
            section.owner_layer is not None
        ), "Section must be associated with a layer."
        assert context.document is not None, "Context must include a document."

        if not self._should_process_section(section):
            return

        # Get the layer configuration for region parsing
        layer_config = self._get_layer_config(section)
        if layer_config is None:
            return

        self._build_role_processor_mapping(layer_config['region_parser_config'])
        self._process_sections_with_processors(context, parent, section, layer_config)
        # self.process_regions(context, parent, section)

    def process_regions(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        self.logger.info("Processing regions section")
        assert context is not None, "Execution context must not be None."
        assert section is not None, "Section must not be None."
        assert parent is not None, "Parent section must not be None."
        assert (
            section.owner_layer is not None
        ), "Section must be associated with a layer."
        assert context.document is not None, "Context must include a document."

        document = context.document
        layer = section.owner_layer
        spans: List[Span] = section.span

        # Regions configuration is expected to be present on the layer (loaded directly from YAML `regions:`)
        region_parser_cfg, regions_cfg, template_fields_repeating = (
            layer.regions_config_raw
        )

        region_parser_cfg = OmegaConf.to_container(region_parser_cfg, resolve=True)
        regions_cfg = OmegaConf.to_container(regions_cfg, resolve=True)

    def _should_process_section(self, section: MatchSection) -> bool:
        """
        Determine if this section should be processed by section processor.

        Args:
            section:  match section

        Returns:
            True if should process, False otherwise
        """
        if section is None:
            return False

        layer = section.owner_layer
        if layer.regions_config_raw is None:
            return False

        region_parser_config, _, _ = layer.regions_config_raw

        return region_parser_config.get('parsing_method') == 'section_processor'

    def _get_layer_config(self, section: MatchSection) -> Optional[Dict]:
        """
        Extract layer configuration for region parsing.

        Args:
            parent: Parent match section

        Returns:
            Layer configuration or None if not found
        """
        layer = section.owner_layer
        if layer is None or layer.regions_config_raw is None:
            return None

        region_parser_config, regions_config, template_fields = layer.regions_config_raw

        return {
            'region_parser_config': region_parser_config,
            'regions_config': regions_config,
        }

    def _build_role_processor_mapping(self, parser_config: Dict) -> None:
        """
        Build a mapping from roles to processor names based on configuration.

        Args:
            parser_config: Parser configuration containing processor definitions
        """
        processors_config = parser_config.get('processors', {})
        for processor_name, processor_config in processors_config.items():
            role = processor_config.get('role')
            if role:
                self._role_to_processor[role] = processor_name

        self.logger.info(f"Built role-to-processor mapping: {self._role_to_processor}")

    def _process_sections_with_processors(
        self,
        context: ExecutionContext,
        parent: MatchSection,
        section: MatchSection,
        layer_config: Dict,
    ) -> None:
        """
        Execute registered processors to parse regions based on their configured roles.

        This method orchestrates the processing pipeline by:
        1. Grouping regions by their assigned processors
        2. Executing each processor with its designated regions
        3. Collecting and validating the results
        4. Integrating parsed regions into the document structure

        Args:
            context: Execution context containing document and processing state
            parent: Parent match section being processed
            layer_config: Layer configuration containing region parser settings

        Raises:
            BadConfigSource: When a required processor is not found in the registry
            ProcessingError: When processor execution fails (in fail_fast mode or after all processors if fail_fast=False)
        """

        region_parser_config = layer_config['region_parser_config']
        regions_config = layer_config['regions_config']

        if not regions_config:
            self.logger.debug("No regions configured for processing")
            return

        regions_by_processor = self._group_regions_by_processor(regions_config)

        if not regions_by_processor:
            self.logger.warning("No processor mappings found for configured regions")
            return

        processed_results: List[StructuredRegion] = []
        processing_errors = []
        processing_stats = {
            "successful": 0,
            "failed": 0,
            "total": len(regions_by_processor),
        }

        for processor_name, processor_regions in regions_by_processor.items():
            processor_function = component_registry.get_region_processor(processor_name)
            if processor_function is None:
                raise BadConfigSource(
                    f"Region processor '{processor_name}' not found in registry"
                )

            try:
                # Execute processor with standardized parameters
                # FIXME: region_parser_config and regions_config need to be corrected, as the semantics are wrong
                processor_result = processor_function(
                    context=context,
                    parent=parent,
                    section=section,
                    region_parser_config=region_parser_config,
                    regions_config=processor_regions,
                )

                # Validate processor output
                self._validate_processor_result(processor_result, processor_name)

                processed_results.append(processor_result)
                processing_stats["successful"] += 1

                self.logger.info(
                    f"Processor parsed '{processor_name}' > {processor_result} regions"
                )

            except Exception as e:
                processing_stats["failed"] += 1
                error_msg = f"Failed to execute processor '{processor_name}': {str(e)}"
                self.logger.error(error_msg, exc_info=True)

                if self.fail_fast:
                    # Re-raise with context for proper error handling upstream
                    raise ProcessingError(error_msg) from e
                else:
                    processing_errors.append((processor_name, str(e)))

        if not self.fail_fast and processing_errors:
            error_summary = "; ".join(
                [f"{name}: {error}" for name, error in processing_errors]
            )
            raise ProcessingError(f"Multiple processors failed: {error_summary}")

        self._integrate_processed_regions(context, processed_results)
        self.logger.info(
            f"Region processing completed: {processing_stats['successful']}/{processing_stats['total']} processors successful"
        )

    def _validate_processor_result(self, result: Any, processor_name: str) -> None:
        """
        Validate processor output conforms to expected structure.

        Args:
            result: Output from processor execution
            processor_name: Name of processor for error reporting

        Raises:
            ProcessingError: When result validation fails
        """
        ...

    def _integrate_processed_regions(
        self, context: ExecutionContext, processed_results: List[StructuredRegion]
    ) -> None:
        """
        Integrate processed regions into the document structure.

        Args:
            context: Execution context containing the document
            processed_results: List of processed region data
        """
        if not processed_results:
            self.logger.debug("No processed regions to integrate")
            return

        document: UnstructuredDocument = context.document
        integration_count = 0

        for region in processed_results:
            try:
                document.insert_region(region)
                integration_count += 1
            except Exception as e:
                self.logger.error(f"Failed to integrate region '{region}': {str(e)}")
                # Continue processing other regions rather than failing completely
                continue
        self.logger.info(
            f"Successfully integrated {integration_count} regions into document"
        )

    def _group_regions_by_processor(
        self, regions_config: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Group regions by their assigned processor based on roles.

        Args:
            regions_config: List of region configurations (from sections)

        Returns:
            Dictionary mapping processor names to their assigned regions
        """
        regions_by_processor = {}

        for region_config in regions_config:
            role = region_config.get('role')
            if not role:
                self.logger.warning(
                    f"No role found for region {region_config.get('title')}"
                )
                continue

            processor_name = self._role_to_processor.get(role)
            if processor_name is None:
                self.logger.warning(f"No processor found for role '{role}'")
                continue

            if processor_name not in regions_by_processor:
                regions_by_processor[processor_name] = []

            regions_by_processor[processor_name].append(region_config)

        return regions_by_processor
