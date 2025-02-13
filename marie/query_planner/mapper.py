import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


@dataclass
class LayoutConfig:
    model_executors: Dict[str, str]
    serverless_executor: Optional[str] = field(default=None)


# 🎯 Base Operator Model
class BaseOperator(BaseModel):
    """Base schema for the operator object."""

    name: str
    type: str = "job"
    uri: str = "s3://bucket/key"
    on: str = Field(..., description="The Executor of the operator.")
    name: str = Field(..., description="The name of the operator.")


# 🎯 Extractor-Specific Operator
class ExtractorOperatorConfig(BaseOperator):
    """Operator structure for EXTRACTOR tasks."""

    pass


# 🎯 Compute-Specific Operator
class ComputeOperatorConfig(BaseOperator):
    """Operator structure for COMPUTE tasks."""

    pass


# 🎯 Merger-Specific Operator
class MergerOperatorConfig(BaseOperator):
    """Operator structure for MERGER tasks."""

    pass


def get_layout_config() -> LayoutConfig:
    parent_config = {
        "model_executors": {
            "qwen_v2_5_vl": "executor_qwen_v2_5_vl_default",
            "qwen_v2_5": "executor_qwen_v2_5_default",
        },
        "serverless_executor": "serverless_executor_default",
    }

    child_config = {
        "model_executors": {
            "qwen_v2_5_vl": "executor_qwen_v2_5_vl",
            "qwen_v2_5": "executor_qwen_v2_5",
        }
    }

    parent_cfg = OmegaConf.create(parent_config)
    child_cfg = OmegaConf.create(child_config)
    merged_cfg = OmegaConf.merge(parent_cfg, child_cfg)

    print("Merged configuration:")
    print(OmegaConf.to_yaml(child_cfg))

    os.exit()

    raw_config = {
        "default": {
            "model_executors": {
                "qwen_v2_5_vl": "executor_qwen_v2_5_vl_default",
                "qwen_v2_5": "executor_qwen_v2_5_default",
            },
            "serverless_executor": "serverless_executor_default",
        },
        "12345": {
            "model_executors": {
                "qwen_v2_5_vl": "executor_qwen_v2_5_vl",
                "qwen_v2_5": "executor_qwen_v2_5",
            }
        },
    }

    yaml_output = yaml.dump(raw_config)
    print(yaml_output)

    schema = OmegaConf.structured(LayoutConfig)
    print(schema)

    return {}
    # config_model = LayoutConfig.model_validate({"data": raw_config})
    # return config_model.data


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
        definition = task.get("definition", {})
        method = definition.get("method", "default")
        endpoint = definition.get("endpoint", "default")
        model_name = definition.get("model_name", "default")
        params = definition.get("params", {})
        executor_endpoint = f"executor://endpoint"

        if node_type is None:
            raise ValueError("Node type is not defined in the task.")

        print(f"Node Type: {node_type}")
        print(f"Definition: {definition}")
        print(f"endpoint: {endpoint}")

        layouts_config = get_layout_config()
        layout_def_base = layouts_config["default"]
        layout_def = layouts_config.get(layout)
        print(f"Layout: {layout_def}")

        print(f"Method: {method}")
        print(f"Params: {params}")

        has_executor = False
        if "://" in endpoint:
            has_executor = True

        if method == 'EXECUTOR_ENDPOINT':
            executor_endpoint = endpoint
        elif method == 'PYTHON_FUNCTION':
            serverless_executor = (
                layout_def.serverless_executor
                if layout_def.serverless_executor
                else "default"
            )
            if has_executor:
                executor_endpoint = endpoint
            else:
                executor_endpoint = f"{serverless_executor}://{endpoint}"
        elif method == 'LLM':
            executor = layout_def.model_executors.get(model_name)
            if not executor:
                raise ValueError(
                    f"Executor not found for model: {model_name}. Available executors: {layout_def.model_executors}"
                )
            executor_endpoint = f"{executor}://{endpoint}"
        elif method == 'NOOP':
            executor_endpoint = "noop://noop"
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

        print('\n\n')

        metadata_class = JobMetadataFactory.get_metadata_class(node_type)
        metadata_obj = metadata_class(
            on=executor_endpoint,
            name=task.get('query_str', 'default'),
        )

        return cls(
            name=node_type,
            metadata=metadata_obj,
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
        return metadata_mapping.get(node_type, BaseOperator)


# ✅ Example Usage
if __name__ == "__main__":
    task_fragment_extractor = {
        "task_id": "067adccc-8316-74a1-8000-202ea146a07b",
        "node_type": "EXTRACTOR",
        "query_str": "19: DOC VALUE EXTRACT",
        "definition": {
            "method": "LLM",
            "params": {"layout": "12345", "extractor": "doc"},
            "endpoint": "extract_field_doc",
            "model_name": "qwen_v2_5_vl",
        },
        "dependencies": ["067adccc-8316-74a1-8000-202ea146a07a"],
    }

    task_fragment_compute = {
        "task_id": "067add2f-8c54-705b-8000-625dcea3d44b",
        "node_type": "COMPUTE",
        "query_str": "18: SEGMENT extracted data",
        "definition": {
            "method": "EXECUTOR_ENDPOINT",
            "params": {"layout": "12345", "function": "segment_data"},
            "endpoint": "extract_executor://segmenter",
        },
        "dependencies": ["067add2f-8c54-705b-8000-625dcea3d44a"],
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

    layout = "12345"

    extractor_metadata = JobMetadata.from_task(task_fragment_extractor, layout)
    compute_metadata = JobMetadata.from_task(task_fragment_compute, layout)

    print("\nExtractor Metadata:")
    print(extractor_metadata.model_dump_json(indent=4))

    print("\nCompute Metadata:")
    print(compute_metadata.model_dump_json(indent=4))
