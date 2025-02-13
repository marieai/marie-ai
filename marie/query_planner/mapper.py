from pydantic import BaseModel, Field


# 🎯 Base Operator Model
class BaseOperator(BaseModel):
    """Base schema for the operator object."""

    # on: str
    # uri: str = "s3://bucket/key"
    name: str
    # type: str = "pipeline"
    # policy: str = "allow_all"
    # ref_id: str = "ref_id_000001"
    # ref_type: str = "doc_type"
    # project_id: str = "project_id_000001"
    # layout: str

    on: str = Field(..., description="The Executor of the operator.")


# 🎯 Extractor-Specific Operator
class ExtractorOperatorConfig(BaseOperator):
    """Operator structure for EXTRACTOR tasks."""

    hello_world: dict = Field(default_factory=lambda: {"enabled": False})


# 🎯 Compute-Specific Operator
class ComputeOperatorConfig(BaseOperator):
    """Operator structure for COMPUTE tasks."""

    computation_mode: str = "fast"
    resource_allocation: str = "standard"


# 🎯 Merger-Specific Operator
class MergerOperatorConfig(BaseOperator):
    """Operator structure for MERGER tasks."""

    merge_strategy: str = "sequential"
    validation_required: bool = True


class JobMetadata(BaseModel):
    """Schema for job metadata."""

    name: str
    action: str = "submit"
    api_key: str = "api_key_000001"
    command: str = "job"
    metadata: BaseOperator
    action_type: str = "command"

    @classmethod
    def from_task(cls, task: dict, layout: str):
        """Creates a JobMetadata instance from a task dictionary."""

        print(f'---------------------------------')
        print(f"Task: {task}")

        node_type = task.get("node_type", "default")
        query_str = task.get("query_str", "default")
        definition = task.get("definition", {})
        method = definition.get("method", "default")
        endpoint = definition.get("endpoint", "default")
        model_name = definition.get("model_name", "default")

        print(f"Node Type: {node_type}")
        print(f"Definition: {definition}")
        print(f"endpoint: {endpoint}")
        executor_endpoint = f"executor://{endpoint}"

        # currently only one layout is supported
        layouts = {
            "12345": {
                "model_endpoint": {
                    "qwen2.5_vl": "extractor_qwen25vl",
                    "qwen2.5": "extractor_qwen25",
                }
            }
        }

        layout = "12345"
        layout_def = layouts.get(layout, {})

        if node_type == 'EXTRACTOR':
            params = definition.get("params", {})
            # type could be doc, field, table, remark, etc.
            print(f"Method: {method}")
            print(f"Params: {params}")
            print(f"model_name: {model_name}")
            executor = layout_def.get("model_endpoint", {}).get(model_name, "default")

            executor_endpoint = f"{executor}://{endpoint}"

        print('\n\n')

        return cls(
            name=node_type,
            metadata=BaseOperator(
                on=executor_endpoint,
                name=task.get('query_str', 'default'),
            ),
        )


# 🎯 Job Metadata Factory
class JobMetadataFactory:
    """
    Factory class to generate different job metadata models based on the node type.
    """

    @staticmethod
    def get_metadata_class(node_type: str):
        """Returns the appropriate metadata model based on node type."""
        metadata_mapping = {
            "EXTRACTOR": ExtractorOperatorConfig,
            "COMPUTE": ComputeOperatorConfig,
            "MERGER": MergerOperatorConfig,
        }
        return metadata_mapping.get(
            node_type, BaseOperator
        )  # Default to base OperatorConfig if unknown


# ✅ Example Usage
if __name__ == "__main__":
    task_fragment_extractor = {
        "task_id": "067ad24b-2f94-7930-8000-d3e26b0adb8e",
        "node_type": "EXTRACTOR",
        "query_str": "20: FIELD VALUE EXTRACT",
        "definition": {
            "method": "LLM",
            "params": {"layout": "12345", "extractor": "field"},
            "endpoint": "extract_field",
            "model_name": "qwen2.5_vl",
        },
        "dependencies": ["067ad24b-2f94-7930-8000-d3e26b0adb8c"],
    }

    task_fragment_compute = {
        "task_id": "067abd09-7e58-7d86-8000-d592660e67c0",
        "node_type": "COMPUTE",
        "query_str": "25: PROCESS DATA",
        "definition": {
            "method": "PYTHON_FUNCTION",
            "params": {"layout": "67890", "function": "process_data"},
        },
        "dependencies": [],
    }

    task_fragment_merger = {
        "task_id": "067abd09-7e58-7d86-8000-d592660e67d0",
        "node_type": "MERGER",
        "query_str": "30: MERGE RESULTS",
        "definition": {
            "method": "NOOP",
            "params": {"layout": "merge_789", "strategy": "sequential"},
        },
        "dependencies": ["067abd09-7e58-7d86-8000-d592660e67c0"],
    }

    # Generate job metadata with a layout parameter
    layout = "fast_layout_123"

    extractor_metadata = JobMetadata.from_task(task_fragment_extractor, layout)
    # compute_metadata = JobMetadata.from_task(task_fragment_compute, layout)
    # merger_metadata = JobMetadata.from_task(task_fragment_merger, layout)
    #
    # Print metadata in JSON format
    print("\nExtractor Metadata:")
    print(extractor_metadata.model_dump_json(indent=4))

    # print("\nCompute Metadata:")
    # print(compute_metadata.model_dump_json(indent=4))
    #
    # print("\nMerger Metadata:")
    # print(merger_metadata.model_dump_json(indent=4))
