import os
import os.path
from typing import TYPE_CHECKING, Any, List, Optional

from marie.constants import __config_dir__
from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.annotators.context_provider import ContextProviderManager
from marie.extract.annotators.util import (
    ascan_and_process_images,
    route_llm_engine,
    scan_and_process_images,
)
from marie.extract.results.result_parser import render_document_markdown
from marie.extract.structures.unstructured_document import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.utils.utils import ensure_exists

if TYPE_CHECKING:
    from marie_kernel.context import RunContext

SYSTEM_PROMPT = ""


def sanitize_path(path: str) -> str:
    """Remove any path traversal attempts from the given path"""
    return os.path.basename(path) if path else None


class LLMAnnotator(DocumentAnnotator):
    """
    LLM Annotator
    """

    def __init__(
        self,
        working_dir: str,
        annotator_conf: dict[str, Any],
        layout_conf: dict[str, Any],
        run_context: Optional["RunContext"] = None,
        **kwargs,
    ):
        """
        Initialize the annotator with a specific value type to extract.
        :param working_dir: Current working directory for the given multi-page document.
        :param annotator_conf: Configuration for the annotator including execution_context.
        :param layout_conf: Layout configuration.
        :param run_context: Optional RunContext for accessing upstream task results.
        """
        super().__init__()
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.silence_exceptions = False
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        # should we merge layout_conf and annotator_conf ?
        self.layout_conf = layout_conf
        self.layout_id = layout_conf.get('layout_id', None)
        if self.layout_id is None:
            raise ValueError("Layout ID is required in the configuration.")

        # configurations from annotator_conf
        self.name = annotator_conf.get('name', None)
        if self.name is None:
            raise ValueError("Annotator name is required in the configuration.")
        self.annotator_type = annotator_conf.get('annotator_type', None)
        self.model_config = annotator_conf.get('model_config', {})

        #  specific configurations from model_config
        self.model_name = self.model_config.get('model_name', None)
        self.prompt_path = self.model_config.get('prompt_path')
        self.system_prompt_text = self.model_config.get('system_prompt_text', None)
        self.top_p = self.model_config.get('top_p', 1.0)
        self.frequency_penalty = self.model_config.get('frequency_penalty', 0)
        self.presence_penalty = self.model_config.get('presence_penalty', 0)
        self.multimodal = self.model_config.get('multimodal', False)
        self.expect_output = self.model_config.get('expect_output', None)

        # Output all parameters for debugging purposes
        self.logger.info(f"Annotator Name: {self.name}")
        self.logger.info(f"Annotator Type: {self.annotator_type}")
        self.logger.info(f"Model Name: {self.model_name}")
        self.logger.info(f"Prompt Path: {self.prompt_path}")
        self.logger.info(f"System Prompt Text: {self.system_prompt_text}")
        self.logger.info(f"Top P: {self.top_p}")
        self.logger.info(f"Frequency Penalty: {self.frequency_penalty}")
        self.logger.info(f"Presence Penalty: {self.presence_penalty}")
        self.logger.info(f"Multimodal: {self.multimodal}")
        self.logger.info(f"Expected Output: {self.expect_output}")

        self.working_dir = working_dir
        self.output_dir = ensure_exists(
            os.path.join(working_dir, "agent-output", self.name)
        )
        self.frames_dir = os.path.join(working_dir, "frames")
        self.logger.info(f'Annotator output dir : {self.output_dir}')

        # Store run_context for direct access to upstream data when needed
        # Usage: self.run_context.get("ANNOTATOR_RESULTS", from_task="tables")
        self.run_context = run_context

        if self.model_name is None:
            raise ValueError("Model name must be provided in the configuration.")

        # TODO : This NEEDS to be moved to a config file
        if self.prompt_path is None and self.system_prompt_text is None:
            raise ValueError(
                "Either prompt_path or system_prompt_text must be provided."
            )

        prompt_dir = kwargs.get("prompt_dir")
        safe_prompt_path = sanitize_path(self.prompt_path) if self.prompt_path else None

        if prompt_dir and safe_prompt_path:
            full_prompt_path = os.path.join(prompt_dir, safe_prompt_path)
        elif safe_prompt_path:
            full_prompt_path = os.path.join(
                __config_dir__,
                "extract",
                f"TID-{self.layout_id}/annotator",
                safe_prompt_path,
            )
        else:
            full_prompt_path = None

        self.prompt_text = self.load_prompt(full_prompt_path)
        self.engine = route_llm_engine(self.model_name, self.multimodal)

        # Get processing mode from annotator config (per-page or per-table)
        self.processing_mode = annotator_conf.get('mode', 'per-table')
        self.logger.info(f"Processing mode: {self.processing_mode}")

        self.context_manager: Optional[ContextProviderManager] = ContextProviderManager(
            run_context=self.run_context,
            annotator_name=self.name,
            mode=self.processing_mode,
        )

        if self.context_manager is not None and self.context_manager.has_providers():
            self.logger.info(
                f"Context providers activated for '{self.name}': "
                f"{[p.__class__.__name__ for p in self.context_manager.providers]}"
            )
        else:
            self.context_manager = None

    @property
    def capabilities(self) -> list:
        return [AnnotatorCapabilities.EXTRACTOR, AnnotatorCapabilities.SEGMENTER]

    def annotate(self, document: UnstructuredDocument, frames: List) -> None:
        """
        Perform value extraction on the given document.

        Upstream task data is available via self.run_context if provided.
        Example: self.run_context.get("ANNOTATOR_RESULTS", from_task="tables")
        """
        self.logger.info(f"Annotating document with {self.name}...")

        # Check if output directory contains results
        if os.listdir(self.output_dir):
            self.logger.info(
                f"Output directory '{self.output_dir}' contains results. Skipping annotation..."
            )
            return

        scan_and_process_images(
            self.frames_dir,
            self.output_dir,
            self.prompt_text,
            document,
            engine=self.engine,
            is_multimodal=self.multimodal,
            expect_output=self.expect_output,
            context_manager=self.context_manager,
        )

    async def aannotate(self, document: UnstructuredDocument, frames: List) -> None:
        """
        Perform value extraction on the given document.

        Upstream task data is available via self.run_context if provided.
        Example: self.run_context.get("ANNOTATOR_RESULTS", from_task="tables")
        """
        self.logger.info(f"Annotating document with {self.name}...")

        # Write context provider debug info to show what variables are being injected
        if self.context_manager:
            import json

            debug_dir = ensure_exists(os.path.join(self.working_dir, "debug"))
            debug_context_path = os.path.join(debug_dir, f"{self.name}_context.json")
            units = self.context_manager.get_processing_units(document)

            if units:
                context_debug = {
                    "annotator": self.name,
                    "total_units": len(units),
                    "providers": [
                        p.__class__.__name__ for p in self.context_manager.providers
                    ],
                    "units": [],
                }
                for unit in units:
                    unit_vars = {}
                    for provider in self.context_manager.providers:
                        unit_vars.update(
                            provider.get_variables(document, unit.page_number, unit)
                        )
                    context_debug["units"].append(
                        {
                            "page": unit.page_number,
                            "index": getattr(unit, "index", None),
                            "variables": unit_vars,
                        }
                    )
                try:
                    with open(debug_context_path, 'w') as f:
                        json.dump(context_debug, f, indent=2, default=str)
                    self.logger.info(f"Context debug written to {debug_context_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to write context debug: {e}")

        # Check if output directory contains results, disable for now
        if os.listdir(self.output_dir):
            self.logger.info(
                f"Output directory '{self.output_dir}' contains results. Skipping annotation..."
            )
            return

        await ascan_and_process_images(
            self.frames_dir,
            self.output_dir,
            self.prompt_text,
            document,
            engine=self.engine,
            is_multimodal=self.multimodal,
            expect_output=self.expect_output,
            context_manager=self.context_manager,
        )

        # self.parse_output(raw_output)

    def parse_output(self, raw_output: str):
        """
        Parse the raw output from value extraction into structured data.
        """
        print("Parsing raw model output...")
        return {}

    def load_prompt(self, prompt_file: str) -> str:
        """Load the prompt text from a file.
        :param prompt_file: Path to the prompt file.
        :return: The prompt text as a string.
        """
        try:
            with open(os.path.expanduser(prompt_file), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            return prompt
        except FileNotFoundError:
            print(f"Unable to find the file: {prompt_file}")
            raise
