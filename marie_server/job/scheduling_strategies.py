from typing import Union, Optional, TYPE_CHECKING

from marie_server.job.placement_group import PlacementGroup


class PlacementGroupSchedulingStrategy:
    """Placement group based scheduling strategy."""

    def __init__(
        self,
        placement_group: "PlacementGroup",
    ):
        self.placement_group = placement_group


class NodeAffinitySchedulingStrategy:
    """Static scheduling strategy used to run a task or actor on a particular node."""

    def __init__(self, node_id: str):
        # This will be removed once we standardize on node id being hex string.
        if not isinstance(node_id, str):
            node_id = node_id.hex()
        self.node_id = node_id


SchedulingStrategyT = Union[
    None,
    str,  # Literal["DEFAULT", "SPREAD"]
    PlacementGroupSchedulingStrategy,
    NodeAffinitySchedulingStrategy,
]
