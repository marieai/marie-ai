import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from marie.constants import __config_dir__, __default_extract_dir__
from marie.query_planner.base import Query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class LayoutConfig:
    """
    Data class for storing layout executor configurations.
    """

    model_executors: Dict[str, str]
    serverless_executor: Optional[str] = field(default=None)


class BaseOperator(BaseModel):
    """
    Base schema for an operator object.
    """

    # this values will overwrtie the parameters in the work item meta
    name: str
    on: str = Field(..., description="Specifies which executor handles this operator.")
    name: str = Field(..., description="The name of the operator.")
    # this parameter dict will be passed in directly without modification
    op_params: Dict[str, Any] = Field(
        default_factory=dict, description="The parameters of the operator."
    )


class ExtractorOperatorConfig(BaseOperator):
    """
    Operator structure for EXTRACTOR tasks.
    """

    pass


class ComputeOperatorConfig(BaseOperator):
    """
    Operator structure for COMPUTE tasks.
    """

    pass


class MergerOperatorConfig(BaseOperator):
    """
    Operator structure for MERGER tasks.
    """

    pass


class BranchOperatorConfig(BaseOperator):
    """
    Operator structure for BRANCH tasks.
    """

    pass


class SwitchOperatorConfig(BaseOperator):
    """
    Operator structure for SWITCH tasks.
    """

    pass


class GuardrailOperatorConfig(BaseOperator):
    """
    Operator structure for GUARDRAIL tasks.
    Quality validation node that evaluates outputs against metrics.
    """

    pass


def has_mapper_config(base_dir: str, layout_name: str) -> bool:
    """
    Check if the mapper configuration file exists for a given layout.

    :param base_dir: The base directory where the configuration files are located.
    :param layout_name: The identifier for the layout configuration.
    :return: True if the configuration file exists, False otherwise.
    """
    layout_cfg_path = os.path.join(base_dir, f"TID-{layout_name}", "mapper.yml")
    return os.path.exists(layout_cfg_path)


def get_mapper_config(base_dir: str, layout_name: str) -> DictConfig:
    """
    Generate a configuration dictionary by merging a base configuration file
    with a layout-specific file. If the layout-specific file doesn't exist,
    falls back to using only the base configuration.

    :param base_dir: The base directory where the configuration files are located.
    :param layout_name: The identifier for the layout configuration.
    :return: A read-only merged configuration as a DictConfig.
    """
    base_cfg_path = os.path.join(base_dir, "base", "mapper.yml")
    layout_cfg_path = os.path.join(base_dir, f"TID-{layout_name}", "mapper.yml")

    if not os.path.exists(base_cfg_path):
        raise FileNotFoundError(
            f"Base mapper configuration not found: {base_cfg_path}. "
            "Ensure the base configuration file exists."
        )

    parent_cfg = OmegaConf.load(base_cfg_path)

    # If layout-specific config exists, merge it with the base config
    # Otherwise, just use the base config
    if os.path.exists(layout_cfg_path):
        layout_cfg = OmegaConf.load(layout_cfg_path)
        merged_conf = OmegaConf.merge(parent_cfg, layout_cfg)
    else:
        logger.debug(
            f"Layout-specific mapper config not found: {layout_cfg_path}. "
            "Using base configuration."
        )
        merged_conf = parent_cfg

    OmegaConf.set_readonly(merged_conf, True)
    return merged_conf


class JobMetadata(BaseModel):
    """
    Schema encapsulating job metadata.
    API KEY will be populated during request creation.
    """

    name: str
    action: str = "submit"
    api_key: str = "api_key_000001"
    command: str = "job"
    metadata: BaseOperator
    action_type: str = "command"

    @classmethod
    def from_task(cls, task: Query, layout: str):
        """
        Construct a JobMetadata instance from a given Query task object.

        :param task: The Query task instance.
        :param layout: Designates which layout configuration to apply.
        :return: A populated JobMetadata object.
        """
        logger.info(
            f"Processing task: {task}",
        )

        # Handle both string and enum types for node_type
        task_type = task.node_type
        if hasattr(task_type, 'value'):
            # It's an enum, get the string value
            task_type = task_type.value
        task_type = str(task_type).upper()

        task_definition = task.definition
        method = task_definition.method if task_definition.method else "NOOP"
        endpoint = (
            task_definition.endpoint if task_definition.endpoint else "noop://noop"
        )
        params = task_definition.params if task_definition.params else {}

        if not endpoint:
            raise ValueError("Endpoint is not defined in the task.")
        if not task_type:
            raise ValueError("Node type is not defined in the task.")

        if not os.path.exists(__default_extract_dir__):
            raise FileNotFoundError(
                f"Base directory does not exist: {__default_extract_dir__}. "
                "Ensure the configuration files are correctly set up."
            )

        layout_conf = get_mapper_config(__default_extract_dir__, layout)
        logger.debug(
            f"Loaded layout config: {layout_conf}",
        )
        logger.debug(f"Task Type: {task_type}, Method: {method}, Params: {params}")

        has_executor = "://" in endpoint

        if method == "EXECUTOR_ENDPOINT":
            executor_endpoint = endpoint
        elif method == "PYTHON_FUNCTION":
            serverless_exec = layout_conf.serverless_executor or "default"
            if has_executor:
                executor_endpoint = (
                    endpoint  # Use original endpoint if it has an executor
                )
            else:
                # Normalize the endpoint path by removing leading slashes
                endpoint_path = endpoint.lstrip('/')
                executor_endpoint = f"{serverless_exec}://{endpoint_path}"
        elif method == "LLM":
            model_name = getattr(task_definition, "model_name", None)
            if not model_name:
                raise ValueError("model_name is required for an LLM method.")
            if has_executor:
                executor_endpoint = endpoint  # Use the original endpoint as-is
            else:
                executor = layout_conf.model_executors.get(model_name)
                if not executor:
                    available = (
                        list(layout_conf.model_executors.keys())
                        if layout_conf.model_executors
                        else []
                    )
                    raise ValueError(
                        f"Executor not found for model '{model_name}' in layout '{layout}'.\n"
                        f"  Available model executors: {available}\n"
                        f"  Task: {task.task_id} ({task.query_str})\n"
                        f"  Config directory: {__default_extract_dir__}\n"
                        f"  To fix: Add '{model_name}' to 'model_executors' in your mapper config, e.g.:\n"
                        f"    model_executors:\n"
                        f"      {model_name}: exec_annotator_{model_name}_default"
                    )
                endpoint_path = endpoint.lstrip('/')
                executor_endpoint = f"{executor}://{endpoint_path}"
            params['model_name'] = model_name
        elif method == "NOOP":
            executor_endpoint = "noop://noop"
        elif method == "BRANCH":
            # BRANCH nodes are control flow nodes that don't execute on an executor
            # They evaluate conditions and mark paths as active or skipped
            executor_endpoint = "branch://control"
        elif method == "SWITCH":
            # SWITCH nodes are control flow nodes for value-based routing
            executor_endpoint = "switch://control"
        elif method == "MERGER_ENHANCED":
            # Enhanced merger that handles skipped branches
            executor_endpoint = endpoint if has_executor else "merger://control"
        elif method == "GUARDRAIL":
            # GUARDRAIL nodes are quality validation nodes that evaluate outputs
            # They don't execute on an executor, evaluation happens in scheduler
            executor_endpoint = "guardrail://control"
        else:
            raise ValueError(f"Unsupported method type: {method}")

        # Determine which operator class to create
        operator_class = JobMetadataFactory.get_metadata_class(task_type)
        operator_instance = operator_class(
            on=executor_endpoint, name=task.query_str, op_params=params
        )

        return cls(
            name=task_type,
            metadata=operator_instance,
        )


class JobMetadataFactory:
    """
    Factory class to generate different job metadata models based on
    the node type.
    """

    @staticmethod
    def get_metadata_class(node_type: str):
        """
        Retrieve the Pydantic model class corresponding to a node type.

        :param node_type: The node type, e.g., "EXTRACTOR", "COMPUTE", "MERGER", "BRANCH", "SWITCH".
        :return: The corresponding class derived from BaseOperator.
        """
        type_mapping = {
            "EXTRACTOR": ExtractorOperatorConfig,
            "COMPUTE": ComputeOperatorConfig,
            "MERGER": MergerOperatorConfig,
            "BRANCH": BranchOperatorConfig,
            "SWITCH": SwitchOperatorConfig,
            "GUARDRAIL": GuardrailOperatorConfig,
        }
        return type_mapping.get(node_type, BaseOperator)


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

    q1 = Query(**task_fragment_extractor)
    print(q1.model_dump())

    extractor_metadata = JobMetadata.from_task(Query(**task_fragment_extractor), layout)
    if False:
        extractor_metadata = JobMetadata.from_task(task_fragment_extractor, layout)
        compute_metadata = JobMetadata.from_task(task_fragment_compute, layout)

        print("\nExtractor Metadata:")
        print(extractor_metadata.model_dump_json(indent=4))

        print("\nCompute Metadata:")
        print(compute_metadata.model_dump_json(indent=4))
