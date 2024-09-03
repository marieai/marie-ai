import warnings
from typing import Dict, List, Optional, Union

bundle_reservation_check = None
BUNDLE_RESOURCE_LABEL = "bundle"


class PlacementGroup:
    """A handle to a placement group."""

    def __init__(self, id: str):
        self.id = id

    @property
    def is_empty(self):
        return self.id.is_nil()

    def __eq__(self, other):
        if not isinstance(other, PlacementGroup):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
