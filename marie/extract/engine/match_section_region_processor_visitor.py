from collections import deque
from typing import Dict, List, Optional

from omegaconf import OmegaConf

from marie.excepts import BadConfigSource
from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection, MatchSectionType
from marie.extract.models.span import Span
from marie.extract.registry import component_registry
from marie.logging_core.logger import MarieLogger


class MatchSectionRegionProcessorVisitor(BaseProcessingVisitor):
    """
    A visitor that uses configurable processors for parsing different types of regions.
    Each processor can specialize in handling specific region types (remarks, codes, notes, etc.).
    """

    def __init__(self, enabled: bool):
        super().__init__(enabled)
        self.logger = MarieLogger(context=self.__class__.__name__)
        self._role_to_processor = {}

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

        print('PROCESSING MATCH SECTION REGION PARSER')
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

        print('region_parser_cfg = ', region_parser_cfg)
        print('regions_cfg = ', regions_cfg)

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
        Process regions using registered processors based on roles.

        Args:
            context: Execution context
            parent: Parent match section
            layer_config: Layer configuration containing region parser settings
        """
        region_parser_config = layer_config['region_parser_config']
        regions_config = layer_config['regions_config']

        parsed_regions = []

        # Group regions by processor based on their roles
        regions_by_processor = self._group_regions_by_processor(regions_config)

        for processor_name, processor_regions in regions_by_processor.items():
            processor_function = component_registry.get_region_processor(processor_name)
            if processor_function is None:
                self.logger.error(
                    f"Region processor '{processor_name}' not found in registry"
                )
                raise BadConfigSource(
                    f"Region processor '{processor_name}' not found in registry"
                )

            try:
                # Call the registered processor
                # FIXME: region_parser_config and regions_config need to be corrected, as the semantics are wrong
                processor_parsed_region = processor_function(
                    context=context,
                    parent=parent,
                    section=section,
                    region_parser_config=region_parser_config,
                    regions_config=processor_regions,
                )

                parsed_regions.extend(processor_parsed_region)
                self.logger.info(
                    f"Processor parsed '{processor_name}' > {processor_parsed_region} regions"
                )

            except Exception as e:
                self.logger.error(
                    f"Error executing processor '{processor_name}': {str(e)}"
                )
                raise

        # Process the results and create regions
        # self._create_regions_from_results(context, parent, parsed_regions)

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

            # Map role to processor name
            print('_role_to_processor = ', role)
            processor_name = self._role_to_processor.get(role)
            if processor_name is None:
                self.logger.warning(f"No processor found for role '{role}'")
                continue

            if processor_name not in regions_by_processor:
                regions_by_processor[processor_name] = []

            regions_by_processor[processor_name].append(region_config)

        return regions_by_processor
