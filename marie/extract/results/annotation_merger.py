import json
from typing import Any, Dict, Hashable, Tuple, Union

from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.concrete_annotations import TypedAnnotation


def _make_hashable(value: Union[str, Dict[str, Any]]) -> Hashable:
    """Convert a value to a hashable form for use as a dict key."""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


class AnnotationMerger:
    """
    Merges duplicate annotations on each line of an UnstructuredDocument.

    If the same (name, value) pair appears more than once with different
    annotation_type, the chosen one is decided by TYPE_PRIORITY (lower is higher priority).
    """

    def __init__(self, type_priority: Dict[str, int]) -> None:
        # Default priorities (lower number = higher priority)
        self._type_priority = type_priority

    def merge(self, doc: UnstructuredDocument) -> None:
        """
        Merges duplicate annotations on each line of the document.

        If the same (name, value) appears more than once (possibly under different
        annotation_type), only one will be kept—chosen by priority.

        Args:
            doc: the UnstructuredDocument whose line.annotations will be deduped.
            type_priority: the type priority dictionary.
        """
        for line in doc.lines:
            anns = line.annotations or []
            if len(anns) <= 1:
                continue

            unique: Dict[Tuple[str, Hashable], TypedAnnotation] = {}
            for ann in anns:
                key = (ann.name, _make_hashable(ann.value))
                if key not in unique:
                    unique[key] = ann
                    continue

                existing = unique[key]
                pr_existing = self._type_priority.get(existing.annotation_type, 99)
                pr_new = self._type_priority.get(ann.annotation_type, 99)
                if pr_new < pr_existing:
                    unique[key] = ann

            line.annotations = list(unique.values())
