from pathlib import Path
from typing import List, Union

from omegaconf import OmegaConf

from marie.excepts import BadConfigSource
from marie.extract.engine.engine import DocumentExtractEngine
from marie.extract.models.base import SelectorSet, TextSelector
from marie.extract.models.definition import (
    FieldCardinality,
    FieldMapping,
    FieldScope,
    Layer,
    Template,
)
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import SubzeroResult
from marie.extract.registry import component_registry
from marie.extract.results.annotation_merger import AnnotationMerger
from marie.extract.schema import ExtractionResult
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.predefined import default_logger as logger


def build_selector_sets(texts: List[str]) -> List[SelectorSet]:
    selectors = [
        TextSelector(tag=f"sel_{i}", text=text, strategy="ANNOTATION")
        for i, text in enumerate(texts)
    ]
    return [SelectorSet(selectors=selectors)]


def build_template(config: OmegaConf) -> Template:
    """
    Builds a template object based on the provided configuration.

    Args:
        config (OmegaConf): Configuration object containing details required to build a template.
                             The configuration should include details about layout ID, layers,
                             fields, tables, and associated mappings.

    Returns:
        Template: A template object constructed using the provided configuration.

    Raises:
        ValueError: If a non-repeating field specified in the layer configuration is not found
                    in the template's non-repeating fields.
    """
    layout_id = str(config.layout_id)
    if layout_id is None:
        raise ValueError("Layout ID is not specified in the configuration.")

    logger.info(f'Building template for layout ID: {layout_id}')

    template = Template(tid=layout_id, name=f"{layout_id}", version=5000)
    layer_name = "layer-main"
    layer_1 = Layer()
    layer_1.layer_name = layer_name

    # TODO: Externalize this mapping
    if layout_id == "1000985":
        layer_1.start_selector_sets = build_selector_sets(
            ["PATIENT NAME", "CLAIM ID", "MEMBER ID", "PATIENT ACCOUNT", "MEMBER"]
        )
    elif layout_id == "121880":  # ORIGINAL 117183
        layer_1.start_selector_sets = build_selector_sets(
            ["PATIENT NAME", "CLAIM NUMBER", "PATIENT ACCOUNT", "CHECK NUMBER"]
        )
    elif layout_id == "122169":  # ORIGINAL 103932 -> 122169
        layer_1.start_selector_sets = build_selector_sets(
            ["CLAIM NUMBER", "PATIENT ACCOUNT", "EMPLOYEE", "PROVIDER TAX ID"]
        )
    else:
        raise ValueError(f"Unsupported layout ID: {layout_id}")

    # layer_1.stop_selector_sets = build_selector_sets(["ISSUED AMT"])
    layer_1.stop_selector_sets = build_selector_sets([])

    layer_conf = config.layers[layer_name]
    template_fields = config.fields
    template_fields_repeating = config.fields.repeating
    template_fields_non_repeating = config.fields.non_repeating

    # Process all field definitions into a unified list
    all_fields: List[FieldMapping] = []

    # Non-repeating fields
    for key, field in layer_conf.non_repeating_fields.items():
        if key not in template_fields_non_repeating:
            raise ValueError(f"Field {key} not found in template fields")
        field_def = template_fields_non_repeating[key]
        annotation_selectors = build_selector_sets(field.annotation_selectors)
        if 'name' not in field_def:
            field_def['name'] = key

        field_mapping = FieldMapping(
            name=key,
            required=False,
            primary=False,
            selector_set=annotation_selectors[
                0
            ],  # There should be only one selector set
            field_def=field_def,
            scope=FieldScope.LAYER,
            cardinality=FieldCardinality.SINGLE,
        )
        all_fields.append(field_mapping)
        layer_1.non_repeating_field_mappings.append(field_mapping)
    layer_1.fields = all_fields

    # Repeating fields - these are fields that can appear multiple times in the document typically in a table

    # Tables are replacing repeating fields, they are more flexible and natural to represent tabular data that contains
    # Headers, Rows and Footers
    layer_1.table_config_raw = (
        layer_conf.tables,
        template_fields_repeating,
    )  # FIXME :  Total hack to get config to work during testing

    layer_1.regions_config_raw = (
        layer_conf.region_parser,
        layer_conf.regions,
        template_fields_repeating,
    )  # FIXME :  Total hack to get config to work during testing

    template.add_layer(layer_1)
    return template


def convert_document_to_structure(
    doc: UnstructuredDocument, conf: OmegaConf, output_dir: Union[Path, str]
) -> SubzeroResult:
    """
    Converts an `UnstructuredDocument` into a structured format using the provided template configuration.

    Args:
        doc (UnstructuredDocument): The document to be processed.
        conf (OmegaConf): Configuration specifying template and extraction parameters.
        output_dir (Union[Path, str]): Directory for output artifacts.

    Returns:
        ExtractionResult: The result of the extraction process.

    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        Exception: For unexpected errors during processing.
    """
    logger.info("Starting conversion of unstructured document to structured format.")

    layout_id = str(conf.layout_id)
    if not layout_id:
        logger.error("Missing layout_id in configuration.")
        raise ValueError("layout_id is required in configuration.")

    logger.info(f"Retrieving template builder for layout ID: {layout_id}")
    builder_fn = component_registry.get_template_builder(layout_id)
    if builder_fn is None:
        logger.error(f"No template builder registered for layout_id={layout_id}")
        raise ValueError(f"No template builder registered for layout_id={layout_id}")

    try:
        template: Template = builder_fn(conf)
    except Exception as e:
        logger.exception(f"Failed to build template for layout_id={layout_id}: {e}")
        raise BadConfigSource(
            f"Failed to build template for layout_id={layout_id}"
        ) from e

    output_dir = Path(output_dir)
    logger.info(f"Output directory set to: {output_dir}")

    doc_id = doc.metadata.get("ref_id", "unknown")
    logger.info(f"Processing document with ID: {doc_id}")

    try:
        annotation_merger = AnnotationMerger(
            OmegaConf.to_container(conf.annotation_config.type_priority)
        )
        annotation_merger.merge(doc)
    except Exception as e:
        logger.exception(f"Annotation merging failed for document {doc_id}: {e}")
        raise

    context = ExecutionContext(
        doc_id=doc_id, template=template, document=doc, output_dir=output_dir, conf=conf
    )

    visitors = conf.get("processing", {}).get("visitors", None)
    if not visitors:
        logger.warning("No visitors specified in configuration. Using core visitors.")

    try:
        results = DocumentExtractEngine(processing_visitors=visitors).match(context)
    except Exception as e:
        logger.exception(f"Document extraction failed for document {doc_id}: {e}")
        raise

    logger.info(f"Document {doc_id} successfully converted to structured format.")
    return results
