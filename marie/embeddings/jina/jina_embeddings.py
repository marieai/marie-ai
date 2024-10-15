import os
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from transformers import AutoModel

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.logger import MarieLogger
from marie.logging_core.profile import TimeContext
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class JinaEmbeddings(EmbeddingsBase):
    def __init__(
        self,
        model_name_or_path: Union[
            str, os.PathLike
        ] = "jinaai/jina-embeddings-v2-base-en",
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 4,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """
        Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents
        https://arxiv.org/abs/2310.19923
        https://huggingface.co/jinaai/jina-embeddings-v2-base-en

        TODO: ADD EXAMPLE AND CODE SNIPPET

        :param model_name_or_path: Directory of a saved model or the name of a public model  from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param show_error: Whether to show errors during inference. If set to False, errors will be logged as debug messages.
        :param kwargs: Additional keyword arguments passed to the model.
        """

        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Embeddings Jina : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]

        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }

        model_name_or_path = ModelRegistry.get(
            model_name_or_path,
            version=None,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )

        assert os.path.exists(model_name_or_path)
        self.logger.info(f"Resolved model : {model_name_or_path}")
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )  # trust_remote_code is needed to use the encode method
        self.model = self.optimize_model(self.model)

    def get_embeddings(
        self, texts: List[str], truncation: bool = None, max_length: int = None
    ) -> EmbeddingsObject:
        with torch.no_grad():
            try:
                embeddings = self.model.encode(
                    texts, batch_size=self.batch_size, truncation=truncation
                )

                result = EmbeddingsObject()
                result.embeddings = embeddings
                result.total_tokens = -1  # len(embeddings[0])

                return result
            except Exception as e:
                self.logger.error(
                    f"Error during inference: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        try:
            with TimeContext("Compiling model", logger=self.logger):
                import torch._dynamo as dynamo

                torch._dynamo.config.verbose = True
                torch._dynamo.config.suppress_errors = True

                # https://dev-discuss.pytorch.org/t/torchinductor-update-4-cpu-backend-started-to-show-promising-performance-boost/874
                model = torch.compile(
                    model, mode="max-autotune", dynamic=True, backend="cudagraphs"
                )
                return model
        except Exception as err:
            raise err
