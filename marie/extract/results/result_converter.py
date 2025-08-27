from pathlib import Path
from typing import Dict, List, Tuple, Union

from omegaconf import OmegaConf

from marie.extract.engine.engine import DocumentExtractEngine
from marie.extract.models.base import SelectorSet, TextSelector
from marie.extract.models.definition import (
    FieldCardinality,
    FieldMapping,
    FieldScope,
    Layer,
    Template,
)
from marie.extract.schema import ExtractionResult
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.concrete_annotations import TypedAnnotation
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


def merge_annotations(doc: UnstructuredDocument) -> None:
    """
    Merges duplicate annotations on each line of the document.

    If the same (name, value) appears more than once (possibly under different
    annotation_type), only one will be kept—chosen by priority.

    Args:
        doc: the UnstructuredDocument whose line.annotations will be deduped.
    """
    # Lower number == higher priority
    # this needs to be configurable in the future
    TYPE_PRIORITY: Dict[str, int] = {"CLAIM": 1, "ANNOTATION": 2}

    for line in doc.lines:
        anns = line.annotations or []
        if len(anns) <= 1:
            continue

        unique: Dict[Tuple[str, str], TypedAnnotation] = {}
        for ann in anns:
            key = (ann.name, ann.value)
            if key not in unique:
                unique[key] = ann
            else:
                # decide which one to keep based on TYPE_PRIORITY
                existing = unique[key]
                # default low priority if missing
                pr_existing = TYPE_PRIORITY.get(existing.annotation_type, 99)
                pr_new = TYPE_PRIORITY.get(ann.annotation_type, 99)
                if pr_new < pr_existing:
                    unique[key] = ann

        # overwrite with the deduped list, preserving priority-chosen annotations
        line.annotations = list(unique.values())


def convert_document_to_structure(
    doc: UnstructuredDocument, conf: OmegaConf, output_dir: Union[Path, str]
) -> ExtractionResult:
    """
    Renders the `UnstructuredDocument` extract basesd on the provided template specification.

    Parameters:
        doc (UnstructuredDocument): The document to render.
        output_dir (Union[Path, str]): The directory to write the output to.

    Returns:
        None
    """
    from marie.extract.models.exec_context import ExecutionContext

    logger.info("Converting unstructured document to structured document")
    # TODO : Add better error handling and validation
    output_dir = Path(output_dir)
    logger.info(f"Writing output to {output_dir}")

    unstructured_meta = doc.metadata
    doc_id = unstructured_meta.get("ref_id", "unknown")

    logger.info(f"Document ID: {doc_id}")
    merge_annotations(doc)

    template = build_template(conf)
    context = ExecutionContext(
        doc_id=doc_id, template=template, document=doc, output_dir=output_dir
    )
    results = DocumentExtractEngine().match(context)

    return results
