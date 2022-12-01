from typing import Dict, Optional, List


from marie.logging.logger import MarieLogger


class TopologyGraph:
    """
    :class TopologyGraph is a class that describes a computational graph of nodes, where each node represents
        a Deployment that needs to be sent requests in the order respecting the path traversal.

    :param graph_description: A dictionary describing the topology of the Deployments. 2 special nodes are expected, the name `start-gateway` and `end-gateway` to
        determine the nodes that receive the very first request and the ones whose response needs to be sent back to the client. All the nodes with no outgoing nodes
        will be considered to be floating, and they will be "flagged" so that the user can ignore their tasks and not await them.

    :param conditions: A dictionary describing which Executors have special conditions to be fullfilled by the `Documents` to be sent to them.
    :param reduce: Reduce requests arriving from multiple needed predecessors, True by default
    """

    def __init__(
        self,
        graph_representation: Dict,
        graph_conditions: Dict = {},
        deployments_metadata: Dict = {},
        deployments_no_reduce: List[str] = [],
        timeout_send: Optional[float] = 1.0,
        retries: Optional[int] = -1,
        logger: Optional[MarieLogger] = None,
        *args,
        **kwargs,
    ):
        ...

    def add_routes(self, request: "DataRequest"):
        """
        Add routes to the DataRequest based on the state of request processing

        :param request: the request to add the routes to
        :return: modified request with added routes
        """
        for node in self._origin_nodes:
            request = node.add_route(request=request)
        return request

    @property
    def origin_nodes(self):
        """
        The list of origin nodes, the one that depend only on the gateway, so all the subgraphs will be born from them and they will
        send to their deployments the request as received by the client.

        :return: A list of nodes
        """
        return self._origin_nodes

    @property
    def all_nodes(self):
        """
        The set of all the nodes inside this Graph

        :return: A list of nodes
        """

        def _get_all_nodes(node, accum, accum_names):
            if node.name not in accum_names:
                accum.append(node)
                accum_names.append(node.name)
            for n in node.outgoing_nodes:
                _get_all_nodes(n, accum, accum_names)
            return accum, accum_names

        nodes = []
        node_names = []
        for origin_node in self.origin_nodes:
            subtree_nodes, subtree_node_names = _get_all_nodes(origin_node, [], [])
            for st_node, st_node_name in zip(subtree_nodes, subtree_node_names):
                if st_node_name not in node_names:
                    nodes.append(st_node)
                    node_names.append(st_node_name)
        return nodes

    def collect_all_results(self):
        """Collect all the results from every node into a single dictionary so that gateway can collect them

        :return: A dictionary of the results
        """
        res = {}
        for node in self.all_nodes:
            if node.result_in_params_returned:
                res.update(node.result_in_params_returned)
        return res
