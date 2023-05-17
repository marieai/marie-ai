from typing import Optional, Any, TypeAlias


class JobDefinition:
    """Defines a Dagster job."""

    def __init__(
        self, *, name: Optional[str] = None, description: Optional[str] = None
    ):
        self.name = name
        self.description = description


InputDefinition: TypeAlias = Any
