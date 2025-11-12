import asyncio
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple

from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.excepts import CUDARuntimeTerminated
from marie.executor.mixin import StorageMixin
from marie.logging_core.logger import MarieLogger
from marie.serve.executors import __dry_run_endpoint__
from marie.utils.server_runtime import setup_storage, setup_toast_events

# Optional NVML
try:
    import pynvml

    _pynvml_exist = True
except ModuleNotFoundError:
    _pynvml_exist = False

# Optional PyTorch
try:
    import torch

    _torch_exist = True
except ModuleNotFoundError:
    _torch_exist = False


class MarieExecutor(Executor, StorageMixin):
    """
    Base executor with:
      - Toast & storage setup
      - Optional periodic GPU health monitor (NVML)
      - Optional PyTorch CUDA probe (dry-run + periodic)
      - Fault marker on fatal errors
      - In-memory last-status cache exposed at /status
      - Termination via CUDARuntimeTerminated (caught by exit_on_exceptions)
    """

    GPU_ERROR_KEYWORDS = (
        "CUDA error",
        "CUDA out of memory",
        "device-side assert",
        "illegal memory access",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUBLAS_STATUS_EXECUTION_FAILED",
        "cudnn",
        "NVML",
        "device lost",
        "HIP error",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("MarieExecutor initialized")
        self.logger.info(f"Kwargs : {kwargs}")

        # ---- Basic setup (toast + storage)
        config = {
            "toast": kwargs.get("toast", {}),
            "storage": kwargs.get("storage", {}),
        }
        self.setup_executor(config)

        # ---- Health monitor config (default off; enable explicitly)
        health_cfg = kwargs.get("health", {}) or {}

        self._nvml_available: bool = _pynvml_exist
        requested_enabled = bool(health_cfg.get("gpu_monitor_enabled", False))
        # monitor runs only if requested and NVML is available
        self._gpu_monitor_enabled: bool = requested_enabled and self._nvml_available

        self._gpu_monitor_interval_s: float = float(health_cfg.get("interval_s", 30))
        self._gpu_monitor_consec_fail: int = int(
            health_cfg.get("consecutive_failures", 3)
        )
        self._gpu_monitor_fail_fast: bool = bool(
            health_cfg.get("fail_fast_on_error", True)
        )

        # Torch probe config
        self._torch_available: bool = _torch_exist
        # default: run torch check only if monitor is enabled and torch is present
        self._torch_check_enabled: bool = bool(
            health_cfg.get(
                "torch_check_enabled",
                self._gpu_monitor_enabled and self._torch_available,
            )
        )
        # run torch probe every N monitor passes
        self._torch_check_every: int = int(health_cfg.get("torch_check_every", 3))
        self._torch_probe_counter: int = 0

        # ---- Runtime fields
        self._gpu_monitor_task: Optional[asyncio.Task] = None
        self._gpu_fail_streak: int = 0
        self._total_failures: int = 0
        self._nvml_ready: bool = False

        # Last observed status (reported via /status)
        self._last_status: Dict[str, Any] = {
            "ts": None,
            "healthy": True,
            "reason": "not probed yet",
            "source": None,
            "exc_class": None,
            "exc_message": None,
            "fail_streak": 0,
            "total_failures": 0,
            "pid": os.getpid(),
            "name": os.getenv("JINA_DEPLOYMENT_NAME", self.__class__.__name__),
            "version": os.getenv("MARIE_BUILD_SHA", None),
            "monitor_enabled": self._gpu_monitor_enabled,
            "nvml_available": self._nvml_available,
            "torch_check_enabled": self._torch_check_enabled,
            "torch_available": self._torch_available,
        }

        if self._gpu_monitor_enabled:
            self._maybe_start_gpu_monitor()
            self.logger.info(
                f"GPU monitor enabled: interval={self._gpu_monitor_interval_s}s, "
                f"consecutive_failures={self._gpu_monitor_consec_fail}, "
                f"torch_check={'on' if self._torch_check_enabled else 'off'}"
            )
        else:
            why = (
                "disabled in config"
                if not requested_enabled
                else "pynvml not available"
            )
            self.logger.info(f"GPU monitor not running ({why})")

    # ---------------------- Setup ----------------------

    def setup_executor(self, config: Dict[str, Any]) -> None:
        """Setup executor with toast events, storage, and asset tracking"""
        setup_toast_events(config.get("toast", {}))
        setup_storage(config.get("storage", {}))

        # Setup storage and asset tracking via StorageMixin
        storage_config = config.get("storage", {})

        print(f'storage_config >>> {storage_config}')
        if storage_config and "psql" in storage_config:
            sconf = storage_config["psql"]
            # Check if asset tracking is enabled in config
            asset_tracking = sconf.get("asset_tracking_enabled", True)
            self.setup_storage(
                storage_enabled=sconf.get("enabled", False),
                storage_conf=sconf,
                asset_tracking_enabled=asset_tracking,
            )
        else:
            self.storage_enabled = False
            self.asset_tracking_enabled = False

        self.logger.info(f"storage_enabled  = {self.storage_enabled}")
        self.logger.info(f"asset_tracking_enabled  = {self.asset_tracking_enabled}")

    # ---------------------- Status / markers ----------------------

    def _set_last_status(
        self,
        *,
        healthy: bool,
        reason: Optional[str] = None,
        source: Optional[str] = None,
        exc: Optional[BaseException] = None,
        fail_streak: Optional[int] = None,
    ):
        self._last_status.update(
            {
                "ts": time.time(),
                "healthy": bool(healthy),
                "reason": reason,
                "source": source,
                "exc_class": exc.__class__.__name__ if exc else None,
                "exc_message": str(exc) if exc else None,
                "fail_streak": (
                    self._gpu_fail_streak if fail_streak is None else fail_streak
                ),
                "pid": os.getpid(),
            }
        )

    def _write_fault_marker(self, reason: str):
        """Atomic fault marker JSON for ops/gateway watchers."""
        try:
            mark_dir = os.getenv("JINA_FAIL_DIR", "/var/run/jina-fail")
            os.makedirs(mark_dir, exist_ok=True)
            name = os.getenv("JINA_DEPLOYMENT_NAME", self.__class__.__name__)
            path = os.path.join(mark_dir, f"{name}.{os.getpid()}.json")
            payload = {
                "ts": time.time(),
                "pid": os.getpid(),
                "name": name,
                "reason": reason,
                "fail_streak": self._gpu_fail_streak,
                "total_failures": self._total_failures,
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            self.logger.exception("Failed to write GPU fault marker")

    # ---------------------- Termination ----------------------

    def _terminate_process(self, reason: str, exc: Optional[BaseException] = None):
        """
        Stop this worker so supervisor/docker can restart it.
        Writes a fault marker and sets status, then raises CUDARuntimeTerminated
        for Jina's exit_on_exceptions to catch.
        """
        self._set_last_status(healthy=False, reason=reason, source="terminate", exc=exc)
        self._write_fault_marker(reason)

        if exc:
            self.logger.error(
                f"Terminating process due to fatal error: {reason} | {exc}",
                exc_info=True,
            )
        else:
            self.logger.error(f"Terminating process due to fatal error: {reason}")

        # Try to cancel monitor task cleanly
        try:
            if self._gpu_monitor_task and not self._gpu_monitor_task.done():
                self._gpu_monitor_task.cancel()
        except Exception:
            pass

        raise CUDARuntimeTerminated

    # ---------------------- NVML lifecycle ----------------------

    def _nvml_init(self) -> Tuple[bool, str]:
        """Ensure NVML is initialized once; return (ready, detail)."""
        if not self._nvml_available:
            return False, "pynvml not available"
        if self._nvml_ready:
            return True, "NVML already initialized"
        try:
            pynvml.nvmlInit()
            self._nvml_ready = True
            return True, "NVML initialized"
        except Exception as e:
            self._nvml_ready = False
            return False, f"NVML init failed: {e!s}"

    def _nvml_shutdown(self):
        if not self._nvml_ready:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        finally:
            self._nvml_ready = False

    # ---------------------- Probes ----------------------

    def _nvml_probe_once(self) -> Tuple[bool, str]:
        """Return (healthy, detail). Conservative: any exception => unhealthy."""
        if not self._nvml_available:
            return True, "NVML unavailable; GPU monitoring skipped"

        ready, detail = self._nvml_init()
        if not ready:
            return False, detail

        try:
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                return False, "no GPU devices reported by NVML"

            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                _ = pynvml.nvmlDeviceGetName(h)
                _ = pynvml.nvmlDeviceGetMemoryInfo(h)  # (total/used/free)

                # Best-effort Xid check (ignore if unsupported)
                try:
                    fv = pynvml.nvmlDeviceGetFieldValues(
                        h, [pynvml.NVML_FI_DEV_XID_ERRORS]
                    )[0]
                    has_val = (
                        getattr(fv, "value_type", None)
                        == pynvml.NVML_VALUE_TYPE_UNSIGNED_INT
                    )
                    if has_val and fv.value_ui > 0:
                        return False, f"Xid errors detected on GPU[{i}]"
                except Exception:
                    pass

            return True, "NVML probe OK"
        except Exception as e:
            # Probe error → drop NVML so next cycle re-inits
            self._nvml_ready = False
            return False, f"NVML probe exception: {e!s}"

    def _torch_probe_once(self) -> Tuple[bool, str]:
        """Quick sanity check that PyTorch can execute a CUDA op."""
        if not (self._torch_check_enabled and self._torch_available):
            return True, "Torch CUDA check skipped"
        try:
            if not torch.cuda.is_available():
                return False, "torch.cuda.is_available() is False"

            cnt = torch.cuda.device_count()
            if cnt <= 0:
                return False, "PyTorch reports zero CUDA devices"

            dev = torch.device("cuda:0")
            a = torch.randn((128, 128), device=dev)
            b = torch.randn((128, 128), device=dev)
            _ = (a @ b).sum().item()
            torch.cuda.synchronize()

            try:
                cudnn_ok = torch.backends.cudnn.is_available()
            except Exception:
                cudnn_ok = None

            return True, f"Torch CUDA OK (devices={cnt}, cudnn={cudnn_ok})"
        except Exception as e:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return False, f"Torch CUDA probe failed: {e.__class__.__name__}: {e}"

    # -------- unified health check (used by dry_run + monitor) --------

    def _health_check_once(
        self,
        *,
        force_torch: bool = False,
        allow_periodic_torch: bool = False,
    ) -> Tuple[bool, str, Literal["nvml", "torch", "combined", "skipped"]]:
        """
        Run one combined health check pass.
        - NVML check (if monitor enabled)
        - Torch check (if force_torch or periodic allowed & counter hits)
        Returns: (ok, detail, source)
        """
        if not self._gpu_monitor_enabled:
            return True, "GPU monitor disabled; health checks skipped", "skipped"

        # 1) NVML first
        ok, detail = self._nvml_probe_once()
        if not ok:
            return False, detail, "nvml"

        # 2) Torch (forced or periodic)
        run_torch = False
        if self._torch_check_enabled:
            if force_torch:
                run_torch = True
            elif allow_periodic_torch:
                self._torch_probe_counter = (self._torch_probe_counter + 1) % max(
                    1, self._torch_check_every
                )
                run_torch = self._torch_probe_counter == 0

        if run_torch:
            tok, tdetail = self._torch_probe_once()
            if not tok:
                return False, tdetail, "torch"
            # both OK
            return True, f"{detail}; {tdetail}", "combined"

        # NVML only, OK
        return True, detail, "nvml"

    # ---------------------- Monitor bootstrap & loop ----------------------

    def _maybe_start_gpu_monitor(self):
        if not self._gpu_monitor_enabled or self._gpu_monitor_task:
            return
        try:
            loop = asyncio.get_running_loop()
            self._gpu_monitor_task = loop.create_task(
                self._gpu_monitor_loop(), name="gpu-monitor"
            )
            self.logger.info(
                f"GPU monitor started: interval={self._gpu_monitor_interval_s}s, "
                f"consecutive_failures={self._gpu_monitor_consec_fail}"
            )
        except RuntimeError:
            self.logger.debug(
                "Event loop not running yet; GPU monitor will start on first request"
            )

    async def _ensure_monitor_started(self):
        if self._gpu_monitor_enabled and not self._gpu_monitor_task:
            self._maybe_start_gpu_monitor()

    async def _gpu_monitor_loop(self):
        # Initial jitter so replicas don't all probe at once
        await asyncio.sleep(0.5 + (os.getpid() % 10) * 0.013)

        base_interval = float(self._gpu_monitor_interval_s)
        backoff_interval = base_interval

        try:
            while True:
                ok, detail, source = self._health_check_once(
                    force_torch=False,
                    allow_periodic_torch=True,
                )

                if ok:
                    if self._gpu_fail_streak:
                        self.logger.warning(
                            f"GPU recovered after {self._gpu_fail_streak} failures: {detail}"
                        )
                    self._gpu_fail_streak = 0
                    self._set_last_status(
                        healthy=True, reason=detail, source=source, fail_streak=0
                    )
                    backoff_interval = base_interval  # reset backoff
                else:
                    self._gpu_fail_streak += 1
                    self._total_failures += 1
                    self._last_status["total_failures"] = self._total_failures
                    self._set_last_status(
                        healthy=False,
                        reason=detail,
                        source=source,
                        fail_streak=self._gpu_fail_streak,
                    )
                    self.logger.error(
                        f"GPU health failed (src={source}, streak={self._gpu_fail_streak}, total={self._total_failures}): {detail}"
                    )

                    # Hard fail at threshold OR fail-fast
                    if (
                        self._gpu_monitor_fail_fast
                        or self._gpu_fail_streak >= self._gpu_monitor_consec_fail
                    ):
                        self._terminate_process(
                            f"GPU monitor hard fail ({source}): {detail}"
                        )
                        await asyncio.sleep(1.0)

                    # Exponential backoff on repeated failures, capped
                    backoff_interval = min(
                        max(base_interval, backoff_interval * 2), 90.0
                    )

                # Jitter to avoid alignment
                sleep_for = backoff_interval + random.random()
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            self.logger.info("GPU monitor cancelled; shutting down")
            raise
        finally:
            self.logger.debug("GPU monitor loop exited")
            try:
                self._nvml_shutdown()
            except Exception:
                pass

    # ---------------------- Endpoints ----------------------

    @requests(on="/")
    async def default_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> DocList[TextDoc]:
        # Start monitor lazily on traffic (if enabled)
        await self._ensure_monitor_started()
        self.logger.debug("Default endpoint called")
        return docs

    @requests(on=__dry_run_endpoint__)
    async def dry_run_func(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        # Skip entirely if GPU monitor is off (CPU-only executor)
        if not self._gpu_monitor_enabled:
            self.logger.debug(
                "Dry run: GPU monitor disabled; skipping NVML/Torch checks"
            )
            return

        await self._ensure_monitor_started()
        self.logger.debug(
            f"Starting dry run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # one unified health pass, but force torch probe here
        try:
            ok, detail, source = self._health_check_once(
                force_torch=True, allow_periodic_torch=False
            )
            if not ok:
                self._set_last_status(
                    healthy=False,
                    reason=detail,
                    source=source,
                    fail_streak=self._gpu_fail_streak,
                )
                self._terminate_process(f"dry-run health failure ({source}): {detail}")
            else:
                self._set_last_status(
                    healthy=True, reason=detail, source=source, fail_streak=0
                )
        except Exception as error:
            self.logger.error(f"Error in dry_run health check: {error}", exc_info=True)
            self._set_last_status(
                healthy=False,
                reason=str(error),
                source="dry_run_exc",
                fail_streak=self._gpu_fail_streak,
            )
            self._terminate_process(f"dry-run error: {error}", error)

    @requests(on="/status")
    async def status_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> DocList[TextDoc]:
        """
        Returns last observed health/exception status as a single JSON string.
        (Kept as TextDoc for wire-compat; switch to structured fields if preferred.)
        """
        # reflect current flags on each read
        self._last_status["monitor_enabled"] = self._gpu_monitor_enabled
        self._last_status["nvml_available"] = self._nvml_available
        self._last_status["torch_check_enabled"] = self._torch_check_enabled
        self._last_status["torch_available"] = self._torch_available
        self._last_status["total_failures"] = self._total_failures

        report = dict(self._last_status)
        payload = json.dumps(report, separators=(",", ":"), ensure_ascii=False)
        return DocList[TextDoc]([TextDoc(text=payload)])
