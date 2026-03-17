import json
import os
import os.path
import shutil
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from marie.constants import __config_dir__
from marie.engine import EngineLM
from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.annotators.context_provider import ContextProviderManager
from marie.extract.annotators.refinement_provider import (
    PassValidationReport,
    ProcessingKey,
    RefinementContextProvider,
    compute_fingerprints,
)
from marie.extract.annotators.util import (
    ascan_and_process_images,
    route_llm_engine,
    scan_and_process_images,
)
from marie.extract.structures.unstructured_document import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.utils.utils import ensure_exists

if TYPE_CHECKING:
    from marie_kernel.context import RunContext

SYSTEM_PROMPT = ""

_SUCCESS_MARKER = "_SUCCESS.yaml"


def sanitize_path(path: str) -> str:
    """Remove any path traversal attempts from the given path"""
    return os.path.basename(path) if path else None


class LLMAnnotator(DocumentAnnotator):
    """LLM Annotator with optional multi-pass refinement."""

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

        # --- Refinement pass config (Step 2a) ---
        self.refine_passes: int = self.model_config.get("refine_passes", 0)
        self.refine_prompt_path = self.model_config.get("refine_prompt_path")
        self.pass_temperatures: Optional[List[float]] = self.model_config.get(
            "pass_temperatures"
        )
        self.pass_models: Optional[List[str]] = self.model_config.get("pass_models")
        self.refinement_validation: dict = self.model_config.get(
            "refinement_validation",
            {
                "require_same_units": True,
                "max_segment_drop_ratio": 0.2,
            },
        )

        if self.refine_passes < 0:
            raise ValueError("refine_passes must be >= 0")
        if (
            self.pass_temperatures
            and len(self.pass_temperatures) < self.refine_passes + 1
        ):
            raise ValueError(
                f"pass_temperatures length ({len(self.pass_temperatures)}) must be "
                f">= refine_passes + 1 ({self.refine_passes + 1})"
            )
        if self.pass_models and len(self.pass_models) < self.refine_passes + 1:
            raise ValueError(
                f"pass_models length ({len(self.pass_models)}) must be "
                f">= refine_passes + 1 ({self.refine_passes + 1})"
            )

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
        self.logger.info(f"Refine Passes: {self.refine_passes}")
        if self.pass_models:
            self.logger.info(f"Pass Models: {self.pass_models}")

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

        # Load refinement prompt if configured
        self.refine_prompt_text: Optional[str] = None
        if self.refine_passes > 0 and self.refine_prompt_path:
            safe_refine = sanitize_path(self.refine_prompt_path)
            if prompt_dir and safe_refine:
                full_refine_path = os.path.join(prompt_dir, safe_refine)
            elif safe_refine:
                full_refine_path = os.path.join(
                    __config_dir__,
                    "extract",
                    f"TID-{self.layout_id}/annotator",
                    safe_refine,
                )
            else:
                full_refine_path = None
            if full_refine_path:
                self.refine_prompt_text = self.load_prompt(full_refine_path)

        # Warn if refinement is enabled but prompt lacks PREVIOUS_EXTRACTION
        if self.refine_passes > 0:
            refine_text = self.refine_prompt_text or self.prompt_text
            if refine_text and "PREVIOUS_EXTRACTION" not in refine_text:
                self.logger.warning(
                    f"Refinement is enabled ({self.refine_passes} passes) but the "
                    "prompt does not contain 'PREVIOUS_EXTRACTION'. "
                    "Prior results will not be injected."
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

    @property
    def capabilities(self) -> list:
        return [AnnotatorCapabilities.EXTRACTOR, AnnotatorCapabilities.SEGMENTER]

    # ------------------------------------------------------------------
    # Skip / completion logic (Step 2b)
    # ------------------------------------------------------------------

    def _has_completed_results(self, live_output_dir: str) -> bool:
        """Check whether a completed run exists in the live output directory.

        A completed run requires:
        - At least one parser-visible result artifact (.json or .md)
        - A ``_SUCCESS.yaml`` marker proving promotion completed

        When refinement is disabled (refine_passes == 0), falls back to the
        original behavior: any file in the directory counts as complete.
        """
        if not os.path.isdir(live_output_dir):
            return False

        entries = os.listdir(live_output_dir)
        if not entries:
            return False

        # Legacy mode: no refinement → any content means done
        if self.refine_passes <= 0:
            return True

        has_marker = _SUCCESS_MARKER in entries
        has_result = any(
            f.endswith((".json", ".md"))
            and not f.startswith("_")
            and os.path.isfile(os.path.join(live_output_dir, f))
            for f in entries
        )
        return has_marker and has_result

    # ------------------------------------------------------------------
    # Path helpers (Step 2c)
    # ------------------------------------------------------------------

    def _scratch_root(self) -> str:
        return os.path.join(self.working_dir, "agent-output", f".{self.name}-refine")

    def _run_root(self, run_id: str) -> str:
        return os.path.join(self._scratch_root(), "runs", run_id)

    def _pass_dir(self, run_id: str, pass_index: int) -> str:
        return os.path.join(self._run_root(run_id), f"pass_{pass_index}")

    def _success_marker_path(self, live_output_dir: str) -> str:
        return os.path.join(live_output_dir, _SUCCESS_MARKER)

    # ------------------------------------------------------------------
    # Per-pass completion params (Step 2d)
    # ------------------------------------------------------------------

    def _completion_params_for_pass(self, pass_index: int) -> dict:
        """Return completion params with optional per-pass temperature override."""
        params = dict(self.completion_params)
        if self.pass_temperatures and pass_index < len(self.pass_temperatures):
            params["temperature"] = self.pass_temperatures[pass_index]
        return params

    # ------------------------------------------------------------------
    # Per-pass engine (Step 2d2)
    # ------------------------------------------------------------------

    def _engine_for_pass(self, pass_index: int) -> EngineLM:
        """Get the LLM engine for a specific pass.

        If ``pass_models`` is configured and has an entry for this pass,
        returns the engine for that model via ``route_llm_engine()`` (cached).
        Otherwise returns ``self.engine``.
        """
        if self.pass_models and pass_index < len(self.pass_models):
            model_name = self.pass_models[pass_index]
            return route_llm_engine(model_name, self.multimodal)
        return self.engine

    # ------------------------------------------------------------------
    # Single extraction helper (Step 2e)
    # ------------------------------------------------------------------

    async def _arun_single_extraction(
        self,
        frames_dir: str,
        output_dir: str,
        prompt_text: str,
        document: UnstructuredDocument,
        completion_params: dict,
        context_manager: Optional[ContextProviderManager],
        engine: Optional[EngineLM] = None,
    ) -> None:
        """Run a single extraction pass into *output_dir*."""
        await ascan_and_process_images(
            frames_dir,
            output_dir,
            prompt_text,
            document,
            engine=engine or self.engine,
            is_multimodal=self.multimodal,
            expect_output=self.expect_output,
            context_manager=context_manager,
            completion_params=completion_params,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mini_batch_size=self.mini_batch_size,
        )

    # ------------------------------------------------------------------
    # Context manager builder (Step 2f)
    # ------------------------------------------------------------------

    def _build_pass_context_manager(
        self,
        refinement_provider: RefinementContextProvider,
    ) -> ContextProviderManager:
        """Build a fresh ``ContextProviderManager`` with the refinement provider.

        Copies existing providers from ``self.context_manager`` (if any) and
        appends the refinement provider.  Never mutates ``self.context_manager``.
        """
        mgr = ContextProviderManager.__new__(ContextProviderManager)
        mgr.providers = []
        mgr.mode = self.processing_mode
        mgr._logger = self.logger

        if self.context_manager is not None:
            mgr.providers = list(self.context_manager.providers)
        mgr.providers.append(refinement_provider)
        return mgr

    # ------------------------------------------------------------------
    # Read pass results (Step 2g)
    # ------------------------------------------------------------------

    def _read_pass_results(self, pass_dir: str) -> Dict[ProcessingKey, str]:
        """Read top-level .json files from *pass_dir* and return raw JSON strings."""
        results: Dict[ProcessingKey, str] = {}
        if not os.path.isdir(pass_dir):
            return results
        for fname in sorted(os.listdir(pass_dir)):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            fpath = os.path.join(pass_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                key = ProcessingKey.from_filename(fname)
            except ValueError:
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                results[key] = f.read()
        return results

    # ------------------------------------------------------------------
    # Validation (Step 2h)
    # ------------------------------------------------------------------

    def _validate_pass_outputs(self, pass_dir: str) -> PassValidationReport:
        """Validate all output files in *pass_dir* and return a report."""
        report = PassValidationReport(json_valid=True)

        if self.expect_output != "json":
            # Non-JSON: permissive report
            files = (
                [
                    f
                    for f in os.listdir(pass_dir)
                    if os.path.isfile(os.path.join(pass_dir, f))
                    and not f.startswith("_")
                    and f.endswith((".json", ".md"))
                ]
                if os.path.isdir(pass_dir)
                else []
            )
            report.file_count = len(files)
            return report

        if not os.path.isdir(pass_dir):
            report.json_valid = False
            report.errors.append(f"Pass directory does not exist: {pass_dir}")
            return report

        for fname in sorted(os.listdir(pass_dir)):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            fpath = os.path.join(pass_dir, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                key = ProcessingKey.from_filename(fname)
            except ValueError:
                continue

            raw = b""
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()
                data = json.loads(raw)
            except (json.JSONDecodeError, OSError) as exc:
                report.json_valid = False
                report.errors.append(f"{fname}: {exc}")
                continue

            report.file_count += 1
            report.processing_keys.add(key)
            report.total_json_size += len(raw)

            # Element count
            extractions = data.get("extractions")
            if isinstance(extractions, list):
                report.total_element_count += len(extractions)
            else:
                report.total_element_count += len(data)

            # Fingerprints
            report.fingerprints_by_key[key] = compute_fingerprints(data)

        return report

    # ------------------------------------------------------------------
    # Report comparison (Step 2i)
    # ------------------------------------------------------------------

    def _compare_pass_reports(
        self,
        previous: PassValidationReport,
        current: PassValidationReport,
    ) -> bool:
        """Return True if *current* pass is acceptable relative to *previous*."""
        if not current.json_valid:
            self.logger.warning("Refinement rejected: invalid JSON")
            return False

        require_same = self.refinement_validation.get("require_same_units", True)
        if require_same and current.processing_keys != previous.processing_keys:
            self.logger.warning(
                "Refinement rejected: processing keys changed "
                f"({len(previous.processing_keys)} -> {len(current.processing_keys)})"
            )
            return False

        max_drop = self.refinement_validation.get("max_segment_drop_ratio", 0.2)
        if previous.total_element_count > 0:
            drop = 1.0 - (current.total_element_count / previous.total_element_count)
            if drop > max_drop:
                self.logger.warning(
                    f"Refinement rejected: element count drop {drop:.1%} > {max_drop:.0%}"
                )
                return False

        # Catastrophic size regression guard
        if previous.total_json_size > 0:
            size_ratio = current.total_json_size / previous.total_json_size
            if size_ratio < 0.3:
                self.logger.warning(
                    f"Refinement rejected: JSON size shrank to {size_ratio:.0%}"
                )
                return False

        # Per-key fingerprint regression
        shared_keys = previous.processing_keys & current.processing_keys
        for key in shared_keys:
            prev_fp = previous.fingerprints_by_key.get(key, set())
            curr_fp = current.fingerprints_by_key.get(key, set())
            if prev_fp:
                retained = len(prev_fp & curr_fp)
                coverage = retained / len(prev_fp)
                if coverage < 0.5:
                    self.logger.warning(
                        f"Refinement rejected: fingerprint coverage for {key} "
                        f"dropped to {coverage:.0%}"
                    )
                    return False

        return True

    # ------------------------------------------------------------------
    # Atomic promotion (Step 2j)
    # ------------------------------------------------------------------

    def _promote_pass_atomically(self, winning_pass_dir: str, run_id: str) -> None:
        """Atomically promote *winning_pass_dir* into the live output directory."""
        live = self.output_dir
        staging = f"{live}.staging.{run_id}"
        backup = f"{live}.backup.{run_id}"

        # 1. Populate staging directory
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)

        promoted_files: list[str] = []
        for fname in os.listdir(winning_pass_dir):
            src = os.path.join(winning_pass_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(staging, fname))
                promoted_files.append(fname)

        # 2. Write success marker
        marker = {
            "run_id": run_id,
            "promoted_pass": os.path.basename(winning_pass_dir),
            "file_count": len(promoted_files),
            "files": sorted(promoted_files),
        }
        with open(os.path.join(staging, _SUCCESS_MARKER), "w") as f:
            yaml.safe_dump(marker, f, default_flow_style=False)

        # 3. Swap: live → backup, staging → live
        try:
            if os.path.exists(live):
                os.rename(live, backup)
            os.rename(staging, live)
        except OSError:
            # Recovery: if staging rename failed, restore backup
            if not os.path.exists(live) and os.path.exists(backup):
                os.rename(backup, live)
            raise

        # 4. Cleanup backup
        if os.path.exists(backup):
            shutil.rmtree(backup, ignore_errors=True)

        self.logger.info(
            f"Promoted {os.path.basename(winning_pass_dir)} "
            f"({len(promoted_files)} files) → {live}"
        )

    # ------------------------------------------------------------------
    # Rerun / recovery helpers (Step 2k)
    # ------------------------------------------------------------------

    def _start_refinement_run(self) -> str:
        """Generate a fresh run_id and create the scratch run directory."""
        run_id = uuid.uuid4().hex[:12]
        os.makedirs(self._run_root(run_id), exist_ok=True)
        return run_id

    def _cleanup_orphaned_runs(self) -> None:
        """Opportunistically remove old scratch runs."""
        scratch = self._scratch_root()
        runs_dir = os.path.join(scratch, "runs")
        if not os.path.isdir(runs_dir):
            return
        for entry in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, entry)
            if os.path.isdir(run_path):
                try:
                    shutil.rmtree(run_path)
                except OSError as e:
                    self.logger.warning(f"Failed to clean orphaned run {entry}: {e}")

    def _recover_or_reset_live_output(self) -> None:
        """Ensure the live output directory is in a clean state.

        If result files exist without a success marker, remove them so a
        new run can start fresh.
        """
        live = self.output_dir
        if not os.path.isdir(live):
            os.makedirs(live, exist_ok=True)
            return

        marker = self._success_marker_path(live)
        if os.path.exists(marker):
            return  # Valid completed state

        # Partial state — clean up
        entries = os.listdir(live)
        if entries:
            self.logger.warning(
                f"Live output dir has {len(entries)} files without {_SUCCESS_MARKER}. "
                "Cleaning for fresh run."
            )
            for entry in entries:
                path = os.path.join(live, entry)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Debug helpers (Step 2n)
    # ------------------------------------------------------------------

    def _write_context_debug(
        self, document: UnstructuredDocument, context_mgr: ContextProviderManager
    ) -> None:
        """Write context-provider debug info to the debug directory."""
        debug_dir = ensure_exists(os.path.join(self.working_dir, "debug"))
        debug_context_path = os.path.join(debug_dir, f"{self.name}_context.json")
        units = context_mgr.get_processing_units(document)

        if not units:
            return

        context_debug: dict = {
            "annotator": self.name,
            "total_units": len(units),
            "providers": [p.__class__.__name__ for p in context_mgr.providers],
            "units": [],
        }
        for unit in units:
            unit_vars: dict = {}
            for provider in context_mgr.providers:
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

    # ------------------------------------------------------------------
    # Refinement orchestration (Step 2l)
    # ------------------------------------------------------------------

    async def _arun_refine_passes(self, document: UnstructuredDocument) -> None:
        """Run initial extraction + N refinement passes, then promote the winner."""
        # Recovery
        self._recover_or_reset_live_output()
        self._cleanup_orphaned_runs()

        run_id = self._start_refinement_run()
        self.logger.info(
            f"Starting refinement run {run_id} "
            f"({self.refine_passes} refinement pass(es))"
        )

        # ---- Pass 0: initial extraction ----
        pass0_dir = ensure_exists(self._pass_dir(run_id, 0))
        pass0_prompt = self.prompt_text.replace("PREVIOUS_EXTRACTION", "")
        pass0_engine = self._engine_for_pass(0)
        pass0_params = self._completion_params_for_pass(0)

        await self._arun_single_extraction(
            frames_dir=self.frames_dir,
            output_dir=pass0_dir,
            prompt_text=pass0_prompt,
            document=document,
            completion_params=pass0_params,
            context_manager=self.context_manager,
            engine=pass0_engine,
        )

        last_good_report = self._validate_pass_outputs(pass0_dir)
        if not last_good_report.json_valid and self.expect_output == "json":
            raise RuntimeError(
                f"Pass 0 produced invalid JSON in run {run_id}: "
                f"{last_good_report.errors}"
            )
        last_good_pass = 0
        self.logger.info(
            f"Pass 0 validated: {last_good_report.file_count} files, "
            f"{last_good_report.total_element_count} elements"
        )

        # ---- Refinement passes 1..N ----
        for i in range(1, self.refine_passes + 1):
            try:
                pass_dir = ensure_exists(self._pass_dir(run_id, i))
                previous_results = self._read_pass_results(
                    self._pass_dir(run_id, last_good_pass)
                )
                provider = RefinementContextProvider(previous_results, self.name)
                pass_ctx = self._build_pass_context_manager(provider)

                prompt = self.refine_prompt_text or self.prompt_text
                engine = self._engine_for_pass(i)
                params = self._completion_params_for_pass(i)

                self.logger.info(f"Running refinement pass {i}/{self.refine_passes}")
                await self._arun_single_extraction(
                    frames_dir=self.frames_dir,
                    output_dir=pass_dir,
                    prompt_text=prompt,
                    document=document,
                    completion_params=params,
                    context_manager=pass_ctx,
                    engine=engine,
                )

                current_report = self._validate_pass_outputs(pass_dir)
                if self._compare_pass_reports(last_good_report, current_report):
                    last_good_pass = i
                    last_good_report = current_report
                    self.logger.info(f"Pass {i} accepted as new best")
                else:
                    self.logger.info(
                        f"Pass {i} rejected; keeping pass {last_good_pass}"
                    )
            except Exception:
                self.logger.warning(
                    f"Pass {i} failed with exception; "
                    f"keeping pass {last_good_pass}",
                    exc_info=True,
                )

        # ---- Atomic promotion ----
        winner_dir = self._pass_dir(run_id, last_good_pass)
        self._promote_pass_atomically(winner_dir, run_id)

        # Write scratch state for debugging
        try:
            state_path = os.path.join(self._scratch_root(), "state.yaml")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w") as f:
                yaml.safe_dump(
                    {
                        "last_run_id": run_id,
                        "promoted_pass": last_good_pass,
                        "total_passes": self.refine_passes + 1,
                    },
                    f,
                    default_flow_style=False,
                )
        except OSError:
            pass  # Non-critical

    # ------------------------------------------------------------------
    # Public annotation methods (Step 2m)
    # ------------------------------------------------------------------

    def annotate(self, document: UnstructuredDocument, frames: List) -> None:
        """
        Perform value extraction on the given document.

        Upstream task data is available via self.run_context if provided.
        Example: self.run_context.get("ANNOTATOR_RESULTS", from_task="tables")
        """
        self.logger.info(f"Annotating document with {self.name}...")

        if self._has_completed_results(self.output_dir):
            self.logger.info(
                f"Output directory '{self.output_dir}' contains completed results. "
                "Skipping annotation..."
            )
            return

        if self.refine_passes > 0:
            from marie.helper import run_async

            run_async(self._arun_refine_passes(document))
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

    async def aannotate(self, document: UnstructuredDocument, frames: List) -> None:
        """
        Perform value extraction on the given document.

        Upstream task data is available via self.run_context if provided.
        Example: self.run_context.get("ANNOTATOR_RESULTS", from_task="tables")
        """
        self.logger.info(f"Annotating document with {self.name}...")

        if self.context_manager:
            self._write_context_debug(document, self.context_manager)

        if self._has_completed_results(self.output_dir):
            self.logger.info(
                f"Output directory '{self.output_dir}' contains completed results. "
                "Skipping annotation..."
            )
            return

        if self.refine_passes > 0:
            await self._arun_refine_passes(document)
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
