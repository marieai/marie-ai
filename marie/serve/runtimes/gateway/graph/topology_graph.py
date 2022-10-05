from typing import Dict, Optional, List


class TopologyGraph:
    def __init__(
        self,
        graph_representation: Dict,
        graph_conditions: Dict = {},
        deployments_disable_reduce: List[str] = [],
        timeout_send: Optional[float] = 1.0,
        *args,
        **kwargs,
    ):
        ...