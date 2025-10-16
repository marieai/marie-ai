import os.path
import time
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from collections import deque
from typing import Optional
from xml.etree.ElementTree import tostring

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    Field,
    MatchSection,
    MatchSectionType,
    SubzeroResult,
)
from marie.extract.results.annotation_highlighter import AnnotationHighlighter
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.logger import MarieLogger


# TODO : This needs to be loaded from G5 dynamically
class MatchSectionRenderingVisitor(BaseProcessingVisitor):
    """
    Renders matched sections and their extracted field values into XML output.
    """

    def __init__(self, enabled: bool):
        super().__init__(enabled)
        self.logger = MarieLogger(context=self.__class__.__name__)

    def visit(self, context: ExecutionContext, parent: SubzeroResult) -> None:
        self.logger.info("Processing MatchSectionRenderingVisitor")
        self.logger.info(f"Rendering to directory {context.output_dir}")

        # highlight_tables(context.document, context.frames, "/tmp/g5")
        # Create root XML element
        root = ET.Element("GrapnelOutput")
        # Add metadata elements
        lbxId = str(context.doc_id or "")
        # Extract the last part of lbxId that comes after the last underscore
        # Example format: PID_1129_7929_0_229725431.tif -> 229725431
        if lbxId and "_" in lbxId:
            lbxId = lbxId.rsplit("_", 1)[-1].replace(".tif", "")
        else:
            lbxId = ""

        ET.SubElement(root, "lbxId").text = lbxId
        ET.SubElement(root, "inputFile").text = context.doc_id or ""
        ET.SubElement(root, "environment").text = "PRODUCTION"
        ET.SubElement(root, "templateid").text = (
            str(context.template.tid) if context.template else ""
        )
        ET.SubElement(root, "version").text = (
            str(context.template.version) if context.template else ""
        )
        # ET.SubElement(root, "duration").text = str(round((datetime.now() - context.start_time).total_seconds(), 3))
        ET.SubElement(root, "duration").text = str(2.5)  # Placeholder for duration

        # Process all content sections and add to XML
        self._process_sections(parent, root)

        xml_string = self._pretty_print_xml(root)

        # Save the XML output
        output_path = os.path.expanduser(
            os.path.join(context.output_dir, f"{context.doc_id}.xml")
        )
        # output_path = os.path.expanduser(os.path.join("~/g5", f"{context.doc_id}.xml"))
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(xml_string)

        self.logger.info(f"XML output written to {output_path}")

    def _process_sections(self, parent: MatchSection, root_element: ET.Element) -> None:
        """
        Recursively process sections and their fields to generate XML structure
        """
        # FIXME: this is a hack and should be externalized
        document_level_field_names = {
            "CHECK_NUMBER",
            "CHECK_AMOUNT",
        }

        # Use BFS to process all sections
        queue = deque([parent])

        while queue:
            current = queue.popleft()
            if (
                current.type == MatchSectionType.WRAPPER
                or not current.matched_non_repeating_fields
                or len(current.matched_non_repeating_fields) == 0
            ):
                queue.extend(current.sections)
                continue
            # Create a remit element for this section
            remit = ET.Element("remits")

            # Add all fields from this section to their respective elements
            for field in current.matched_non_repeating_fields:
                field_elem = self._convert_field(field)

                if field.field_name in document_level_field_names:
                    root_element.append(field_elem)
                else:
                    remit.append(field_elem)

            # Process all matched_field_rows for this section
            if current.matched_field_rows is not None:
                for row in current.matched_field_rows:
                    service_lines = ET.SubElement(
                        remit, "serviceLines"
                    )  # Directly create <serviceLines> for each row
                    # our structure is foobared as there is no container for the fields  but rather they are added one at a  time
                    for field in row.fields:
                        self._convert_field(
                            field,
                            parent_element=service_lines,
                            default_date_format="",
                            default_column_name="",
                        )

            # Only add the remit element if it has sub elements
            if len(remit) > 0:
                root_element.append(remit)

            # Add child sections to the queue
            queue.extend(current.sections)

    def _convert_field(
        self,
        field: Field,
        parent_element: ET.Element = None,
        default_date_format: str = None,
        default_column_name: str = None,
    ) -> ET.Element:
        """
        Converts a Field object into its XML representation.

        This method generates an XML element that represents a given `Field` object. The generated
        element contains all relevant attributes of the `Field` object, including its name, type,
        position, and other metadata. Optional attributes such as `date_format`, `column_name`,
        `uuid`, and `reference_uuid` are also added when available. The newly created XML element
        can optionally be appended to a parent XML element.

        Parameters:
            field (Field): The Field object to be converted into an XML element.
            parent_element (ET.Element, optional): An optional parent XML element to which the
                generated XML element will be appended. Defaults to None.
            default_date_format (str, optional): An optional default date format to use if the
                `Field` object does not have a specific `date_format`. Defaults to None.
            default_column_name (str, optional): An optional default column name to use if the
                `Field` object does not have a specific `column_name`. Defaults to None.

        Returns:
            ET.Element: The XML representation of the Field object.
        """
        field_elem = ET.Element("fields")
        if parent_element is not None:
            parent_element.append(field_elem)

        ET.SubElement(field_elem, "fieldName").text = field.field_name or ""
        ET.SubElement(field_elem, "fieldType").text = field.field_type or "ALPHA"
        ET.SubElement(field_elem, "isRequired").text = str(field.is_required).lower()
        ET.SubElement(field_elem, "value").text = field.value or ""
        ET.SubElement(field_elem, "x").text = str(field.x)
        ET.SubElement(field_elem, "y").text = str(field.y)
        ET.SubElement(field_elem, "width").text = str(field.width)
        ET.SubElement(field_elem, "height").text = str(field.height)
        ET.SubElement(field_elem, "page").text = str(field.page)
        ET.SubElement(field_elem, "xdpi").text = str(field.xdpi)
        ET.SubElement(field_elem, "ydpi").text = str(field.ydpi)
        ET.SubElement(field_elem, "confidence").text = str(field.confidence)
        ET.SubElement(field_elem, "scrubbed").text = str(field.scrubbed).lower()

        # Add optional field attributes
        if field.date_format or default_date_format is not None:
            date_format = ET.SubElement(field_elem, "dateFormat")
            ET.SubElement(date_format, "formatString").text = (
                field.date_format or default_date_format
            )
        if field.column_name or default_column_name is not None:
            ET.SubElement(field_elem, "columnName").text = (
                field.column_name or default_column_name
            )
        if field.uuid:
            ET.SubElement(field_elem, "uuid").text = field.uuid
        if field.reference_uuid:
            ET.SubElement(field_elem, "referenceUuid").text = field.reference_uuid

        if field.value_original is not None:
            ET.SubElement(field_elem, "valueOriginal").text = field.value_original

        return field_elem

    def _pretty_print_xml(self, element: ET.Element) -> str:
        """
        Pretty print an XML element, ensuring all values are serialized as strings.
        """

        def ensure_text_is_string(elem):
            """
            Recursively ensure all .text and .tail of an element and its children
            are converted to strings, if they are not None.
            """
            if elem.text is not None:
                elem.text = str(elem.text)
            if elem.tail is not None:
                elem.tail = str(elem.tail)
            for child in elem:
                ensure_text_is_string(child)

        ensure_text_is_string(element)
        rough_string = tostring(element, 'utf-8').decode('utf-8')

        dom = minidom.parseString(rough_string)
        return dom.toprettyxml(indent="    ")
        # return rough_string


def highlight_tables(doc: UnstructuredDocument, frames: list, output_dir: str):
    def tables_only(ann, _line) -> bool:
        try:
            return ann.annotation_type == "TABLE"
        except AttributeError:
            return False

    table_colors = {
        "TABLE_START": (0, 200, 0, 96),
        "TABLE_END": (200, 0, 0, 96),
    }

    def label_for(ann, _line) -> Optional[str]:
        try:
            if ann.name:
                return str(ann.name)
        except AttributeError:
            pass
        return None

    highlighter = AnnotationHighlighter(
        color_by="name",
        colors=table_colors,
        default_color=(255, 165, 0, 80),
        label_fn=label_for,
        filter_fn=tables_only,
        use_annotation_bboxes=True,
        font_size=14,
        output_name_fn=lambda page_id: f"{page_id + 1}-tables.png",
    )

    highlighter.highlight_document(doc, frames, output_dir)
