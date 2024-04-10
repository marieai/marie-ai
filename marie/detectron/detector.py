import os
from typing import Callable

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from torch import nn
from torch._C._profiler import ProfilerActivity
from torch.nn import Module
from torch.profiler import profile

from marie.logging.logger import MarieLogger
from marie.logging.profile import TimeContext
from marie.models.utils import torch_gc
from marie.utils.types import strtobool


class OptimizedDetectronPredictor:
    """
    Optimized version of the detectron2 predictor.
    """

    def __init__(self, cfg, half_precision=True):
        """
        Initialize the model with the given config.

        :param cfg: the detectron2 config
        :param half_precision:   whether to use half precision or not (default: True) will only work on CUDA
        """
        self.logger = MarieLogger(self.__class__.__name__)
        self.profiler_enabled = strtobool(
            os.environ.get("MARIE_PROFILER_ENABLED", False)
        )
        self.half_precision = (
            True if half_precision and cfg.MODEL.DEVICE == "cuda" else False
        )

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        # self.model = self.optimize_model(self.model)

        if self.half_precision:
            self.logger.info("Detectron half precision enabled")
            self.model = self.model.half()
            self.model.to(self.cfg.MODEL.DEVICE)

        for param in self.model.parameters():
            param.grad = None

        self.model.eval()

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        return self.invoke_model(original_image, raise_oom=False)

    def invoke_model(self, original_image, raise_oom=False):
        """
        Invoke the model with the given image.

        :param original_image:  an image of shape (H, W, C) (in BGR order).
        :param raise_oom:  whether to raise OOM exception or not
        :return:  the output of the model for one image only.
        """
        if raise_oom:
            self.logger.warning("OOM detected, clearing cache and retrying")

        with torch.inference_mode():
            with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
                try:
                    # clear cache
                    # Apply pre-processing to image.
                    if self.input_format == "RGB":
                        # whether the model expects BGR inputs or RGB
                        original_image = original_image[:, :, ::-1]
                    height, width = original_image.shape[:2]
                    image = self.aug.get_transform(original_image).apply_image(
                        original_image
                    )
                    image = torch.as_tensor(
                        image.astype("float32").transpose(2, 0, 1),
                        device=self.cfg.MODEL.DEVICE,
                    )
                    if self.half_precision:
                        image = image.half()
                    # image.to(self.cfg.MODEL.DEVICE)

                    inputs = {"image": image, "height": height, "width": width}
                    if self.profiler_enabled:
                        # ensure that output directory exists
                        os.makedirs(
                            os.path.expanduser("~/tmp/cuda-profiler"), exist_ok=True
                        )

                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            with_stack=True,
                            profile_memory=True,
                        ) as prof:
                            predictions = self.model([inputs])[0]

                        # Print aggregated stats
                        print(
                            prof.key_averages(group_by_stack_n=5).table(
                                sort_by="self_cuda_time_total", row_limit=2
                            )
                        )
                        prof.export_stacks(
                            os.path.expanduser(
                                "~/tmp/cuda-profiler/profiler_stacks.txt"
                            ),
                            "self_cuda_time_total",
                        )
                        prof.export_chrome_trace(
                            os.path.expanduser("~/tmp/cuda-profiler/trace.json")
                        )
                    else:
                        predictions = self.model([inputs])[0]

                    del inputs
                    return predictions
                except RuntimeError as e:
                    if "out of memory" in str(e) and not raise_oom:
                        print("| WARNING: ran out of memory")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                        return self.invoke_model(original_image, raise_oom=True)
                    else:
                        raise e
                finally:
                    torch_gc()

    def optimize_model(self, model: nn.Module) -> Callable | Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        try:
            with TimeContext("Compiling model", logger=self.logger):
                import torch._dynamo as dynamo
                import torchvision.models as models

                torch._dynamo.config.verbose = True
                torch._dynamo.config.suppress_errors = True
                # torch.backends.cudnn.benchmark = True
                # https://dev-discuss.pytorch.org/t/torchinductor-update-4-cpu-backend-started-to-show-promising-performance-boost/874
                # model = torch.compile(model )
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                return model
        except Exception as err:
            self.logger.warning(f"Model compile not supported: {err}")
            return model
