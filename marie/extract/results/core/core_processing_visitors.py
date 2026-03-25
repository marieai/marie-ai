"""Register record-backed MatchSection visitors with the component registry.

These visitors are automatically available in the engine bootstrap pipeline.
They can also be referenced by name in ``visitor_names`` configuration.
"""

from marie.extract.engine.record_backed_match_section_builder_visitor import (
    RecordBackedMatchSectionBuilderVisitor,
)
from marie.extract.engine.record_backed_match_section_population_visitor import (
    RecordBackedMatchSectionPopulationVisitor,
)
from marie.extract.registry import component_registry

component_registry.register_processing_visitor(
    "RecordBackedMatchSectionBuilderVisitor"
)(RecordBackedMatchSectionBuilderVisitor)

component_registry.register_processing_visitor(
    "RecordBackedMatchSectionPopulationVisitor"
)(RecordBackedMatchSectionPopulationVisitor)
