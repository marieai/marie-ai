import json
import os
import os.path
import re
from typing import Any, List

from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.structures.unstructured_document import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.utils.utils import ensure_exists


class RegexAnnotator(DocumentAnnotator):
    """
    Regex-based Annotator
    """

    def __init__(
        self,
        working_dir: str,
        annotator_conf: dict[str, Any],
        layout_conf: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__()
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        self.layout_conf = layout_conf
        self.layout_id = layout_conf.get('layout_id', None)
        if not self.layout_id:
            raise ValueError("Layout ID is required in the configuration.")

        self.name = annotator_conf.get('name')
        if not self.name:
            raise ValueError("Annotator name is required in the configuration.")
        self.annotator_type = annotator_conf.get('annotator_type')

        self.regex_patterns = annotator_conf.get('regex', [])
        if not self.regex_patterns:
            raise ValueError("Regex patterns must be provided.")

        self.working_dir = working_dir
        self.output_dir = ensure_exists(
            os.path.join(working_dir, "agent-output", self.name)
        )

    @property
    def capabilities(self) -> list:
        return [AnnotatorCapabilities.EXTRACTOR]

    def annotate(self, document: UnstructuredDocument, frames: List) -> None:
        self.logger.info(f"Annotating document with {self.name}...")

        if os.listdir(self.output_dir):
            self.logger.info(
                f"Output directory '{self.output_dir}' contains results. Skipping annotation..."
            )
            return

        all_text = document.get_all_text()  # You may have a more precise API call

        extracted = []
        for rule in self.regex_patterns:
            if not rule.get("enabled", True):
                continue

            pattern = re.compile(rule["regex"])
            matches = pattern.findall(all_text)
            for match in matches:
                extracted.append(
                    {
                        "name": rule["name"],
                        "value": (
                            match[rule["group"] - 1]
                            if isinstance(match, tuple)
                            else match
                        ),
                        "type": rule.get("type", "unknown"),
                        "confidence": rule.get("confidence", 1.0),
                    }
                )

        output_path = os.path.join(self.output_dir, "results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2)

        self.logger.info(f"Extraction complete. Results saved to: {output_path}")
