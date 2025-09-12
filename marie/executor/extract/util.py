import os
from typing import Optional

from omegaconf import OmegaConf

from marie.logging_core.predefined import default_logger as logger
from marie.utils.utils import ensure_exists


def load_layout_config(base_dir: str, layout_dir: str, debug_config=False) -> OmegaConf:
    # TODO : THIS NEEDS TO BE CONFIGURABLE
    layout_dir = os.path.expanduser(layout_dir)
    base_dir = os.path.expanduser(base_dir)

    # root config
    base_cfg_path = os.path.join(base_dir, "base-config.yml")
    field_cfg_path = os.path.join(base_dir, "field-config.yml")
    # layout config
    layout_cfg_path = os.path.join(layout_dir, "config.yml")
    base_cfg = OmegaConf.load(base_cfg_path)
    field_cfg = OmegaConf.load(field_cfg_path)
    specific_cfg = OmegaConf.load(layout_cfg_path)
    merged_conf = OmegaConf.merge(field_cfg, base_cfg, specific_cfg)

    # Keys can come in few formats
    # - PATIENT NAME
    # - PATIENT    NAME
    # - PATIENT_NAME
    # for that we will normalize spaces into underscores (_)

    merged_conf.grounding = {
        k: [entry.replace(" ", "_") for entry in v]
        for k, v in merged_conf.grounding.items()
    }

    if debug_config:
        with open("merged_config_output.yml", "w") as yaml_file:
            yaml_file.write(OmegaConf.to_yaml(merged_conf))
            print(OmegaConf.to_yaml(merged_conf))

    return merged_conf


def layout_config(root_config_dir: str, layout_id: str) -> OmegaConf:
    base_dir = os.path.expanduser(os.path.join(root_config_dir, "base"))
    layout_dir = os.path.expanduser(
        os.path.join(root_config_dir, f"TID-{layout_id}/annotator")
    )

    logger.info(f"Layout ID : {layout_id}")
    logger.info(f"Base dir : {base_dir}")
    logger.info(f"Layout dir : {layout_dir}")

    cfg = load_layout_config(base_dir, layout_dir)
    logger.debug(f"Layout config : {cfg}")
    return cfg


def setup_table_directories(
    working_dir: str, annotator_name: Optional[str] = None
) -> tuple:
    """Setup required directories for table annotation."""
    output_dir = ensure_exists(os.path.join(working_dir, "agent-output"))
    htables_output_dir = os.path.join(output_dir, "highlighted_tables")
    table_src_dir = os.path.join(output_dir, "tables")
    table_annotated_dir = os.path.join(output_dir, "table_annotated")
    table_annotated_fragments_dir = os.path.join(
        output_dir, "table_annotated", "fragments"
    )

    # specific to the annotator that is being used
    annotator_output_dir = None
    if annotator_name:
        annotator_output_dir = ensure_exists(os.path.join(output_dir, annotator_name))

    ensure_exists(htables_output_dir)
    ensure_exists(table_annotated_dir)
    ensure_exists(table_annotated_fragments_dir)

    return (
        htables_output_dir,
        table_src_dir,
        table_annotated_dir,
        table_annotated_fragments_dir,
        annotator_output_dir,
    )
