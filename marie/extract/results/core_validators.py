import logging
import os

from omegaconf import OmegaConf

from marie.extract.results.registry import register_validator
from marie.extract.structures import UnstructuredDocument


@register_validator("noop")
def validate_noop(
    doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
) -> None:
    """
    No-op validator for annotations. This is a placeholder and does not perform any action.
    """
    logging.info("No-op validator for annotations called. No action taken.")
    # This function is intentionally left empty to serve as a placeholder.
    pass
