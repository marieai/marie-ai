# Python
from typing import Optional, Tuple

from marie.extract.structures.structured_region import SectionRole

SemanticRole = str  # free-form semantic label


def normalize_role(value: object) -> Tuple[SectionRole, Optional[SemanticRole]]:
    """
    Accepts:
      - SectionRole instance
      - string (enum name/value or arbitrary)
    Returns:
      (layout_role: SectionRole, semantic_role: Optional[str])
    Known enum names/values map to SectionRole directly; unknown strings become semantic_role.
    """
    if isinstance(value, SectionRole):
        return value, None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return SectionRole.UNKNOWN, None

        # Try enum member name (e.g., "MAIN", "CONTEXT_ABOVE")
        try:
            return SectionRole[s.upper()], None
        except KeyError:
            pass

        # Try enum value (e.g., "main", "context_above")
        for member in SectionRole:
            if member.value == s.lower():
                return member, None

        # Otherwise treat as semantic role
        return SectionRole.UNKNOWN, s

    return SectionRole.UNKNOWN, None
