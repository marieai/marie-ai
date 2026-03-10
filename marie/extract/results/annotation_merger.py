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

    An optional second pass resolves same-name-different-value conflicts using
    a configurable strategy ('none', 'longest_value', or 'type_priority').
    """

    def __init__(
        self,
        type_priority: Dict[str, int],
        conflict_resolution: str = "none",
    ) -> None:
        # Default priorities (lower number = higher priority)
        self._type_priority = type_priority
        self._conflict_resolution = conflict_resolution

    def merge(self, doc: UnstructuredDocument) -> None:
        """
        Merges duplicate annotations on each line of the document.

        Pass 1: If the same (name, value) appears more than once (possibly under
        different annotation_type), only one will be kept—chosen by priority.

        Pass 2 (optional): If the same name appears with different values, resolve
        the conflict using the configured strategy.

        Args:
            doc: the UnstructuredDocument whose line.annotations will be deduped.
        """
        for line in doc.lines:
            anns = line.annotations or []
            if len(anns) <= 1:
                continue

            # === Pass 1: existing (name, value) dedup ===
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

            # === Pass 2: name-level conflict resolution ===
            if self._conflict_resolution != "none":
                by_name: Dict[str, TypedAnnotation] = {}
                for ann in unique.values():
                    if ann.name not in by_name:
                        by_name[ann.name] = ann
                        continue
                    existing = by_name[ann.name]
                    if self._conflict_resolution == "longest_value":
                        existing_len = len(str(existing.value)) if existing.value else 0
                        new_len = len(str(ann.value)) if ann.value else 0
                        if new_len > existing_len:
                            by_name[ann.name] = ann
                    elif self._conflict_resolution == "type_priority":
                        pr_existing = self._type_priority.get(
                            existing.annotation_type, 99
                        )
                        pr_new = self._type_priority.get(ann.annotation_type, 99)
                        if pr_new < pr_existing:
                            by_name[ann.name] = ann
                line.annotations = list(by_name.values())
            else:
                line.annotations = list(unique.values())
