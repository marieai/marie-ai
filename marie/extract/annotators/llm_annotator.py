import glob
import json
import os
import os.path
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
from marie.utils.types import to_bool
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
        self.annotator_conf = annotator_conf
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
        self.temperature = self.model_config.get('temperature', 0.0)
        self.extra_body = self.model_config.get('extra_body', None)
        self.min_pixels = self.model_config.get('min_pixels', 512 * 28 * 28)
        self.max_pixels = self.model_config.get('max_pixels', 2048 * 28 * 28)
        self.mini_batch_size = self.model_config.get('mini_batch_size', 16)

        # Build completion_params dict from config
        self.completion_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.extra_body is not None:
            self.completion_params["extra_body"] = self.extra_body

        self.mm_processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        # Output all parameters for debugging purposes
        self.logger.info(f"Annotator Name: {self.name}")
        self.logger.info(f"Annotator Type: {self.annotator_type}")
        self.logger.info(f"Model Name: {self.model_name}")
        self.logger.info(f"Prompt Path: {self.prompt_path}")
        self.logger.info(f"System Prompt Text: {self.system_prompt_text}")
        self.logger.info(f"Temperature: {self.temperature}")
        self.logger.info(f"Top P: {self.top_p}")
        self.logger.info(f"Frequency Penalty: {self.frequency_penalty}")
        self.logger.info(f"Presence Penalty: {self.presence_penalty}")
        self.logger.info(f"Extra Body: {self.extra_body}")
        self.logger.info(f"Min Pixels: {self.min_pixels}")
        self.logger.info(f"Max Pixels: {self.max_pixels}")
        self.logger.info(f"Mini Batch Size: {self.mini_batch_size}")
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
        full_prompt_path = self._resolve_prompt_path(safe_prompt_path, prompt_dir)
        self.prompt_text = self.load_prompt(full_prompt_path)
        self.engine = route_llm_engine(self.model_name, self.multimodal)

        # Output transform: a dotted path to a callable applied in-place to each
        # output file after LLM inference.
        # Signature: fn(data: dict, annotator_conf: dict) -> dict
        self.output_transform: Optional[str] = annotator_conf.get(
            'output_transform', None
        )

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

    def _resolve_prompt_path(
        self, safe_prompt_path: Optional[str], prompt_dir: Optional[str]
    ) -> Optional[str]:
        """Resolve prompt path with TID-specific -> base fallback.

        Resolution order:
          1. TID-specific: {root}/TID-{layout_id}/annotator/{prompt}
          2. Base fallback: {root}/base/{prompt}
          3. Returns the path as-is if neither exists (load_prompt will raise)
        """
        if not safe_prompt_path:
            return None

        if prompt_dir:
            full_path = os.path.join(prompt_dir, safe_prompt_path)
            if not os.path.exists(full_path):
                # prompt_dir is .../TID-X/annotator/ -> go up two levels to reach base/
                base_dir = os.path.join(
                    os.path.dirname(os.path.dirname(prompt_dir)), "base"
                )
                fallback = os.path.join(base_dir, safe_prompt_path)
                if os.path.exists(fallback):
                    self.logger.info(
                        f"Prompt '{safe_prompt_path}' not in TID dir, "
                        f"using base: {fallback}"
                    )
                    return fallback
            return full_path

        # Production executor path: derive from __config_dir__
        full_path = os.path.join(
            __config_dir__,
            "extract",
            f"TID-{self.layout_id}/annotator",
            safe_prompt_path,
        )
        if not os.path.exists(full_path):
            fallback = os.path.join(__config_dir__, "extract", "base", safe_prompt_path)
            if os.path.exists(fallback):
                self.logger.info(
                    f"Prompt '{safe_prompt_path}' not in TID dir, "
                    f"using base: {fallback}"
                )
                return fallback
        return full_path

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
            purge = to_bool(os.environ.get("MARIE_PURGE_OUTPUT"))
            if purge:
                self.logger.info(
                    f"MARIE_PURGE_OUTPUT is set — clearing output directory '{self.output_dir}'"
                )
                shutil.rmtree(self.output_dir)
                os.makedirs(self.output_dir, exist_ok=True)
            else:
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
            completion_params=self.completion_params,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mini_batch_size=self.mini_batch_size,
        )

        if self.output_transform:
            self._apply_output_transform()

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
            completion_params=self.completion_params,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mini_batch_size=self.mini_batch_size,
        )

        if self.output_transform:
            self._apply_output_transform()

    def _apply_output_transform(self) -> None:
        """Apply the configured output_transform function in-place to each JSON output file.

        The transform is a dotted Python path (e.g.
        ``.extract.core.transforms.normalize_labels``)
        resolved via importlib.  Signature::

            fn(data: dict, annotator_conf: dict) -> dict
        """
        import importlib

        json_files = sorted(glob.glob(os.path.join(self.output_dir, "*.json")))
        if not json_files:
            return

        # Resolve the callable once
        transform_path = self.output_transform
        try:
            module_name, func_name = transform_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            transform_fn = getattr(module, func_name)
        except (ImportError, AttributeError, ValueError) as e:
            self.logger.error(
                f"Could not resolve output_transform '{transform_path}': {e}"
            )
            return

        self.logger.info(
            f"Applying output_transform '{transform_path}' to {len(json_files)} file(s)"
        )

        # Preserve original files before any transforms are applied
        pre_transform_dir = os.path.join(self.output_dir, "pre-transform")
        os.makedirs(pre_transform_dir, exist_ok=True)
        for filepath in json_files:
            shutil.copy2(
                filepath, os.path.join(pre_transform_dir, os.path.basename(filepath))
            )
        self.logger.info(
            f"Saved {len(json_files)} original file(s) to {pre_transform_dir}"
        )

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Skipping {filepath}: {e}")
                continue

            try:
                data = transform_fn(data, self.annotator_conf)
            except Exception as e:
                self.logger.error(
                    f"output_transform '{transform_path}' failed on {filepath}: {e}"
                )
                continue

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

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
