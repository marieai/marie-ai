import traceback
from collections import defaultdict, deque
from io import StringIO
from pprint import pprint
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from marie.job.job_manager import generate_job_id, increment_uuid7str
from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    PythonFunctionQueryDefinition,
    Query,
    QueryPlan,
    QueryPlanRegistry,
    QueryType,
)
from marie.query_planner.mapper import JobMetadata


class NoAliasDumper(yaml.Dumper):  # NoAliasDumper promoted outside the function
    def ignore_aliases(self, data):
        return True


def plan_to_json(plan: QueryPlan, output_path: str = "query_plan_pretty.json") -> str:
    """
    Dumps the QueryPlan to a JSON file with enhanced formatting for better readability.
    :param plan:
    :param output_path:
    :return:
    """

    plan_to_json_str = plan.model_dump_json()
    with open(output_path, "w") as json_file:
        json_file.write(plan_to_json_str)
    return plan_to_json_str


def plan_to_yaml(plan: QueryPlan, output_path: str = "query_plan_pretty.yaml") -> str:
    """
    Dumps the QueryPlan to a YAML file with enhanced formatting for better readability.

    :param plan: The QueryPlan to dump.
    :param output_path: The file path to save the YAML file (default: 'query_plan_pretty.yaml').
    :return: The YAML content as a string.
    """
    # Convert the QueryPlan model to a dictionary
    plan_dict = plan.model_dump()

    # Create an in-memory string buffer for YAML output
    yaml_buffer = StringIO()

    # Dump the YAML content to the string buffer
    yaml.dump(
        plan_dict,
        yaml_buffer,
        Dumper=NoAliasDumper,
        default_flow_style=False,  # Use block format for better readability
        sort_keys=False,  # Preserve key order
    )

    # Get the YAML string from the buffer
    yaml_str = yaml_buffer.getvalue()

    # Write the YAML string to the output file
    with open(output_path, "w") as yaml_file:
        yaml_file.write(yaml_str)

    return yaml_str  # Return the YAML string


def print_sorted_nodes(sorted_nodes: list[str], plan: QueryPlan) -> None:
    """
    Pretty prints the sorted nodes, with their descriptions, and the query plan in sorted order.

    :param sorted_nodes: List of nodes in topologically sorted order.
    :param plan: The QueryPlan object.
    """
    if True:
        return

    print("\n" + "=" * 60)
    print("Topologically Sorted Nodes:")
    print("=" * 60)
    print(sorted_nodes)

    print("\n" + "=" * 60)
    print("Query Plan (Sorted Order):")
    print("=" * 60)

    # Create a full dependency graph for each node
    full_dependencies = {}
    for node in plan.nodes:
        visited = set()
        queue = deque(node.dependencies)
        all_deps = set()

        while queue:
            dep_id = queue.popleft()
            if dep_id not in visited:
                visited.add(dep_id)
                all_deps.add(dep_id)
                # Add the dependencies of the current dependency to the queue
                dep_node = next((n for n in plan.nodes if n.task_id == dep_id), None)
                if dep_node:
                    queue.extend(dep_node.dependencies)

        full_dependencies[node.task_id] = list(all_deps)

    node_dict = {node.task_id: node for node in plan.nodes}

    for node_id in sorted_nodes:
        node = node_dict.get(node_id)
        if node:
            full_dep = full_dependencies[node_id]
            print(
                f"Query ID: {node.task_id} | Description: {node.query_str} | dependencies: {node.dependencies} | Full Dependencies: {full_dep}"
            )


def topological_sort(plan: QueryPlan) -> list:
    """
    Perform topological sorting on the QueryPlan based on node dependencies.

    :param plan: The QueryPlan to sort.
    :return: A list of node IDs in topologically sorted order.
    :raises ValueError: If a cycle is detected in the graph. (Graph is not a DAG)
    """
    # Step 1: Build the adjacency list and in-degree count
    adjacency_list = defaultdict(list)
    in_degree = defaultdict(int)

    # Initialize all nodes in in-degree
    for node in plan.nodes:
        in_degree[node.task_id] = 0

    # Fill adjacency list and in-degree based on dependencies
    for node in plan.nodes:
        for dependency_id in node.dependencies:
            adjacency_list[dependency_id].append(node.task_id)
            in_degree[node.task_id] += 1

    # Step 2: Find all nodes with zero in-degree (no dependencies)
    zero_in_degree_queue = deque(
        [node_id for node_id, degree in in_degree.items() if degree == 0]
    )

    # Step 3: Perform topological sorting
    sorted_nodes = []
    while zero_in_degree_queue:
        node_id = zero_in_degree_queue.popleft()
        sorted_nodes.append(node_id)

        # Reduce in-degree of dependent nodes
        for dependent_node in adjacency_list[node_id]:
            in_degree[dependent_node] -= 1
            if in_degree[dependent_node] == 0:
                zero_in_degree_queue.append(dependent_node)

    # Step 4: Check for cycles (graph isn't a DAG if all nodes are not sorted)
    if len(sorted_nodes) != len(plan.nodes):
        raise ValueError(
            "The QueryPlan graph has a cycle and cannot be topologically sorted."
        )

    return sorted_nodes


def compute_job_levels(sorted_nodes: list[str], plan: QueryPlan) -> dict[str, int]:
    """
    Given a list of node IDs in topological order and the original QueryPlan,
    compute a job_level for each node representing its distance from the nearest root.

    :param sorted_nodes: Node IDs in topological order.
    :param plan: The original QueryPlan with dependency information.
    :return: A dict mapping node_id to its level (int).
    """
    # Build a map from node_id to its dependencies for quick lookup
    dependency_map = {node.task_id: node.dependencies for node in plan.nodes}

    # Initialize all levels to 0
    job_level = {node_id: 0 for node_id in sorted_nodes}

    # Iterate in topological order, so dependencies are processed first
    for node_id in sorted_nodes:
        deps = dependency_map.get(node_id, []) or []
        if deps:
            # level = max level of dependencies + 1
            max_dep_level = max(job_level.get(dep, 0) for dep in deps)
            job_level[node_id] = max_dep_level + 1

    return job_level


def _load_query_planner(planner_name: str) -> Callable:
    """
    Dynamically load the query planner based on the planner name.

    :param planner_name: The name of the query planner to load.
    :return: A callable query planner function.
    :raise ValueError: If the query planner name is invalid.
    """
    import os

    logger.debug(f"Loading query planner: {planner_name}")
    try:
        return QueryPlanRegistry.get(planner_name)
    except ValueError as e:
        logger.error(f"Error loading query planner: {e}")
        if os.getenv("MARIE_DEBUG_QUERY_PLANNER", "false").lower() == "true":
            logger.error(
                f"Available planners: {list(QueryPlanRegistry.list_planners())}"
            )
        raise


def query_planner(planner_info: PlannerInfo) -> QueryPlan:
    query_planner_name = planner_info.name
    try:
        query_planner_fn = _load_query_planner(query_planner_name)
    except ValueError as e:
        logger.error(f"Error loading query planner: {e}")
        raise

    plan = query_planner_fn(planner_info)
    if not plan:
        error_message = (
            f"Query planner '{query_planner_name}' returned an invalid plan."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Technically DAG can have multiple roots, but we want to enforce
    # a single root node as that will be used as the entry point
    strict = True
    plan = ensure_single_entry_point(plan, planner_info, strict)
    logger.debug(f"Query planning completed successfully for: {query_planner_name}")
    return plan


def ensure_single_entry_point(
    query_plan: QueryPlan, planner_info: PlannerInfo, strict: bool = False
) -> QueryPlan:
    """
    Ensure the query plan has a single entry point by creating a synthetic root node
    that connects to all natural root nodes in the DAG.

    Args:
        query_plan:  QueryPlan object to update with a synthetic root node
        planner_info: PlannerInfo object containing base_id and current_id

    Returns:
        Tuple of (updated query plan, incremented current_id)
        :param strict:
    """
    nodes = query_plan.nodes
    base_job_id = planner_info.base_id
    current_id = planner_info.current_id

    referenced_nodes = set()
    for node in nodes:
        referenced_nodes.update(node.dependencies)

    # Find natural root nodes (nodes that aren't dependencies of any other node)
    natural_roots = [node for node in nodes if node.task_id not in referenced_nodes]

    strict = False
    if strict and len(natural_roots) != 1:
        error_message = (
            f"Query plan must have exactly one root node, found {len(natural_roots)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if len(natural_roots) == 1:
        return query_plan

    current_id += 1
    root = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: ROOT",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    for natural_root in natural_roots:
        natural_root.dependencies.append(root.task_id)

    query_plan.nodes.insert(0, root)
    planner_info.current_id = current_id

    return query_plan


def query_planner_xyz(frame_count: int, layout: str) -> QueryPlan:
    """
    Plan a structured query execution graph for document annotation, adding SEGMENT, FIELD & TABLE Extraction, and COLLATOR steps.
    :param frame_count: Number of frames to process.
    :param layout: The layout of the document.
    :return: The QueryPlan for the structured query execution graph.

    Example:
    START -> ROOT (Annotator) -> Frame1 -> Frame2 -> MERGE -> COLLATOR -> END

    """
    base_job_id = generate_job_id()

    current_id = 0
    # Root node for the entire process
    root = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    current_id += 1

    annotator_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: Document annotator",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    current_id += 1

    steps = []
    frame_joiners = []
    frame_annotators = []

    for i in range(frame_count):
        # Frame segmentation step
        frame_annotator = Query(
            task_id=f"{increment_uuid7str(base_job_id, current_id)}",
            query_str=f"{current_id}: Annotate frame {i} ROIs",
            dependencies=[annotator_node.task_id],  # No dependency
            node_type=QueryType.COMPUTE,
            definition=NoopQueryDefinition(params={"layout": layout}),
        )
        current_id += 1
        frame_annotators.append(frame_annotator)

        step_start = Query(
            task_id=f"{increment_uuid7str(base_job_id, current_id)}",
            query_str=f"{current_id}: ROI_START for frame {i}",
            dependencies=[frame_annotator.task_id],  # Now dependent on segmenter
            node_type=QueryType.COMPUTE,
            definition=LlmQueryDefinition(
                model_name="qwen_v2_5_vl",
                endpoint="annotator/roi",
                params={"layout": layout, "roi": "start"},
            ),
        )
        current_id += 1

        step_end = Query(
            task_id=f"{increment_uuid7str(base_job_id, current_id)}",
            query_str=f"{current_id}: ROI_END for frame {i}",
            dependencies=[frame_annotator.task_id],  # Now dependent on segmenter
            node_type=QueryType.COMPUTE,
            definition=LlmQueryDefinition(
                model_name="qwen_v2_5_vl",
                endpoint="annotator/roi",
                params={"layout": layout, "roi": "end"},
            ),
        )
        current_id += 1

        step_relation = Query(
            task_id=f"{increment_uuid7str(base_job_id, current_id)}",
            query_str=f"{current_id}: ROI_RELATION for frame {i}",
            dependencies=[frame_annotator.task_id],  # Now dependent on segmenter
            node_type=QueryType.COMPUTE,
            definition=LlmQueryDefinition(
                model_name="qwen_v2_5_vl",
                endpoint="annotator/roi",
                params={"layout": layout, "roi": "relation"},
            ),
        )
        current_id += 1

        steps.extend([frame_annotator, step_start, step_end, step_relation])

        # Frame-level joiner now depends on all frame annotations
        joiner = Query(
            task_id=f"{increment_uuid7str(base_job_id, current_id)}",
            query_str=f"{current_id}: Merge frame {i}",
            dependencies=[step_start.task_id, step_end.task_id, step_relation.task_id],
            node_type=QueryType.MERGER,
            definition=NoopQueryDefinition(params={'layout': layout}),
        )
        current_id += 1

        frame_joiners.append(joiner)

    # Global joiner (Merging results from all frames)
    global_joiner = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: Merge all frames",
        dependencies=[
            joiner.task_id for joiner in frame_joiners
        ],  # Depends on all frame joiners
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={'layout': layout}),
    )
    current_id += 1

    # Additional Processing Steps after Global Joiner
    segment_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: SEGMENT extracted data",
        dependencies=[global_joiner.task_id],  # Depends on global merged results
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract_executor://segmenter",
            params={'layout': layout, 'function': 'segment_data'},
        ),
    )
    current_id += 1

    document_value_extract_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: DOC VALUE EXTRACT",
        dependencies=[segment_node.task_id],  # Depends on segmentation step
        node_type=QueryType.EXTRACTOR,
        definition=LlmQueryDefinition(
            model_name="qwen_v2_5_vl",
            endpoint="extract/field_doc",
            params={'layout': layout, 'extractor': 'doc'},
        ),
    )
    current_id += 1

    field_value_extract_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: FIELD VALUE EXTRACT",
        dependencies=[segment_node.task_id],  # Depends on segmentation step
        node_type=QueryType.EXTRACTOR,
        definition=LlmQueryDefinition(
            model_name="qwen_v2_5_vl",
            endpoint="extract/field",
            params={'layout': layout, 'extractor': 'field'},
        ),
    )
    current_id += 1

    table_value_extract_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: TABLE VALUE EXTRACT",
        dependencies=[segment_node.task_id],  # Depends on segmentation step
        node_type=QueryType.EXTRACTOR,
        definition=LlmQueryDefinition(
            model_name="qwen_v2_5_vl",
            endpoint="extract/table",
            params={'layout': layout, 'extractor': 'table'},
        ),
    )
    current_id += 1

    # EXTRACT node now depends on both field & table extraction steps
    extract_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: EXTRACT insights",
        dependencies=[
            document_value_extract_node.task_id,
            field_value_extract_node.task_id,
            table_value_extract_node.task_id,
        ],
        node_type=QueryType.MERGER,
        definition=PythonFunctionQueryDefinition(
            endpoint="evaluator",
            params={'layout': layout, 'function': 'extract_insights'},
        ),
    )
    current_id += 1

    collator_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: COLLATOR: Compile structured output",
        dependencies=[extract_node.task_id],  # Depends on extraction step
        node_type=QueryType.MERGER,
        definition=PythonFunctionQueryDefinition(
            endpoint="evaluator", params={'layout': layout, 'function': 'data_collator'}
        ),
    )
    current_id += 1

    end_node = Query(
        task_id=f"{increment_uuid7str(base_job_id, current_id)}",
        query_str=f"{current_id}: END",
        dependencies=[collator_node.task_id],  # Depends on extraction step
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={'layout': layout}),
    )

    return QueryPlan(
        nodes=[root]
        + [annotator_node]
        + steps
        + frame_joiners
        + [
            global_joiner,
            segment_node,
            document_value_extract_node,
            field_value_extract_node,
            table_value_extract_node,
            extract_node,
            collator_node,
        ]
        + [end_node]
    )


def visualize_query_plan_graph(plan: QueryPlan, output_path="query_plan_graph.png"):
    """
    Prints and saves a consistent and predictable graph visualization of the query plan
    as a Directed Acyclic Graph (DAG).

    :param plan: The QueryPlan to visualize.
    :param output_path: The file path to save the graph image (default: 'query_plan_graph.png').
    """

    # Create a directed graph (DiGraph)
    graph = nx.DiGraph()

    if False:
        return

    # Add nodes and edges to the graph in a consistent order
    for node in sorted(plan.nodes, key=lambda x: x.task_id):  # Sort nodes by ID
        graph.add_node(
            node.task_id,
            label=node.query_str,
            type=node.node_type.name,
        )
        for dependency in sorted(
            node.dependencies
        ):  # Sort dependencies to ensure order
            graph.add_edge(
                dependency,
                node.task_id,  # Create directed edges
                flow="data_flow",  # Custom edge attribute to denote data flow
            )

    # Validate that the graph is a DAG
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "The query plan graph is not a DAG! Check for circular dependencies."
        )

    # Node color based on node type
    node_colors = []
    for _, data in graph.nodes(data=True):
        node_type = data["type"]
        if node_type == "COMPUTE":
            node_colors.append("#6baed6")  # Light blue for compute nodes
        elif node_type == "MERGER":
            node_colors.append("#74c476")  # Green for joiners
        elif node_type == "SEGMENTER":
            node_colors.append("#fd8d3c")  # Orange for segmenters
        elif node_type == "EXTRACTOR":
            node_colors.append("#fdae6b")  # Light orange for extractors
        else:
            node_colors.append("#d9d9d9")  # Default gray

    # Use a consistent hierarchical layout (Graphviz) for reproducibility
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")  # Top-down layout
    except ImportError:
        # If Graphviz is not available, fall back to spring_layout with a fixed seed
        pos = nx.spring_layout(graph, seed=42)  # Setting a seed for consistent results

    # Start plotting
    plt.figure(figsize=(14, 12))  # Larger figure size for clarity
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=2500, alpha=0.95
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrowstyle="-|>",
        arrowsize=20,  # Larger arrow size for better visibility
        edge_color="slategray",
        width=2,
        connectionstyle="arc3,rad=0.1",  # Slightly curved edges
    )

    # Add node labels
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: f"{data['label']}" for node, data in graph.nodes(data=True)},
        font_size=9,
        font_color="black",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="lightgray"),
    )

    # Add legend for node types
    legend_items = {
        "Root/Single Question": "#6baed6",
        "Mergers (MERGER)": "#74c476",
        "Segmenters (SEGMENTER)": "#fd8d3c",
        "Extractors (EXTRACTOR)": "#fdae6b",
    }
    legend_patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markersize=10,
            markerfacecolor=color,
        )
        for label, color in legend_items.items()
    ]
    plt.legend(
        handles=legend_patches, loc="upper left", title="Node Types", frameon=True
    )

    # Set plot title and layout
    plt.title("Query Plan Graph: Directed Acyclic Graph (DAG)", fontsize=16)
    plt.axis("off")
    plt.tight_layout(pad=1.0)

    # Save the graph locally
    plt.savefig(output_path, format="png", dpi=300)  # Save with high resolution

    plt.show()


def nx_to_mermaid(graph: nx.DiGraph) -> str:
    lines = ["```mermaid", "flowchart TD"]

    for source, target in graph.edges():
        lines.append(f"    {source} --> {target}")

    lines.append("```")
    return "\n".join(lines)


def print_query_plan(plan: QueryPlan, layout: str) -> None:
    """Print the query plan in a human-readable format."""

    executors = []
    for node in plan.nodes:
        print("Node:", node.task_id)
        try:
            meta = JobMetadata.from_task(node, layout)
            executors.append(meta.metadata.on)
            print(meta.model_dump())
        except Exception as e:
            traceback.print_exc()
            print(f"Error creating metadata for node {node.task_id}: {e}")
    print("Executors:", executors)
    for exec in executors:
        print(f"Executor: {exec}")


if __name__ == "__main__":
    question = "Annotate documents with named entities."
    frame_count = 3
    layout = "extract"
    planner_info = PlannerInfo(name="extract", base_id=generate_job_id())
    plan = query_planner(planner_info)
    pprint(plan.model_dump())
    visualize_query_plan_graph(plan)
    # yml_str = plan_to_yaml(plan)
    # pprint(yml_str)

    json_serialized = plan.model_dump()
    reconstructed_plan = QueryPlan(**json_serialized)
    sorted_nodes = topological_sort(plan)
    print_sorted_nodes(sorted_nodes, plan)

    print_query_plan(plan, layout)
