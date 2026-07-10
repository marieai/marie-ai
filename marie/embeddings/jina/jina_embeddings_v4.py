"""jina-embeddings-v4 for unified text + image embeddings.

This module provides multimodal embedding capabilities using jina-embeddings-v4,
which embeds both text and images into the same vector space for unified retrieval.

Key features:
- Single embedding space for text AND images
- Task-specific adapters: retrieval, text-matching, code
- Query/passage prompts for retrieval optimization
- Matryoshka support: truncate to 128/256/512/1024/2048 dimensions
- 32K token context window

Reference: https://huggingface.co/jinaai/jina-embeddings-v4
"""

import os
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from torch import nn

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class JinaEmbeddingsV4(EmbeddingsBase):
    """Jina Embeddings v4 for unified text + image embeddings.

    This implementation supports the jina-embeddings-v4 model which provides
    a single embedding space for both text and images, making it ideal for
    multimodal RAG applications.

    Key differences from JinaEmbeddings (v2):
    - Multimodal: embeds text AND images in the same space
    - API: encode_text() / encode_image() instead of encode()
    - Task adapters: retrieval, text-matching, code
    - Query/passage prompts for retrieval
    - Matryoshka: can truncate to smaller dimensions

    Example:
        ```python
        embeddings = JinaEmbeddingsV4(
            model_name_or_path="jinaai/jina-embeddings-v4",
            task="retrieval",
            truncate_dim=1024,  # Use smaller dimension
        )

        # Embed text
        text_result = embeddings.get_embeddings(
            ["What is machine learning?"],
            is_query=True,  # Use query prompt
        )

        # Embed images (same embedding space!)
        image_result = embeddings.get_image_embeddings(["path/to/image.jpg"])

        # Can directly compare text and image embeddings
        similarity = cosine_similarity(text_result.embeddings[0], image_result.embeddings[0])
        ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike] = "jinaai/jina-embeddings-v4",
        model_version: Optional[str] = None,
        task: Literal["retrieval", "text-matching", "code"] = "retrieval",
        truncate_dim: int = 2048,
        use_gpu: bool = True,
        batch_size: int = 4,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Initialize JinaEmbeddingsV4.

        Args:
            model_name_or_path: Model name from HuggingFace or local path.
            model_version: Model version tag/branch/commit.
            task: Task adapter to use. Options:
                - "retrieval": For search/RAG (recommended for most cases)
                - "text-matching": For semantic similarity
                - "code": For code search
            truncate_dim: Output dimension. Matryoshka supports:
                128, 256, 512, 1024, 2048 (default).
                Smaller = faster but less accurate.
            use_gpu: Whether to use GPU acceleration.
            batch_size: Batch size for encoding.
            use_auth_token: HuggingFace auth token for private models.
            devices: Specific devices to use.
            show_error: Whether to show detailed errors.
            torch_dtype: Torch dtype (e.g., torch.float16 for memory efficiency).
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Initializing JinaEmbeddingsV4: {model_name_or_path}")

        self.task = task
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.show_error = show_error
        self._torch_dtype = torch_dtype or torch.float16

        # Validate truncate_dim
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            self.logger.warning(
                f"truncate_dim {truncate_dim} not in standard Matryoshka dimensions {valid_dims}. "
                f"Using closest valid dimension."
            )
            self.truncate_dim = min(valid_dims, key=lambda x: abs(x - truncate_dim))

        # Device setup
        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices not supported, using first device %s",
                resolved_devices[0],
            )
        self.device = resolved_devices[0]

        # Resolve model path via registry
        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }

        resolved_path = ModelRegistry.get(
            model_name_or_path,
            version=model_version,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )

        self.logger.info(f"Resolved model path: {resolved_path}")

        # Load model
        self.model = self._load_model(resolved_path)
        self.logger.info(
            f"JinaEmbeddingsV4 initialized: task={task}, dim={self.truncate_dim}, device={self.device}"
        )

    @staticmethod
    def _shim_transformers_compat() -> None:
        """Backfill transformers APIs the jina-v4 remote code
        (``qwen2_5_vl.py``) imports but that postdate the installed
        transformers checkout (4.50.0.dev0). Every definition is a faithful
        copy of the upstream implementation; the guards make each shim a
        no-op once transformers is upgraded."""
        import functools
        import sys
        import types

        import transformers
        import transformers.modeling_flash_attention_utils as fa_utils
        import transformers.modeling_rope_utils as rope_utils
        import transformers.utils as tf_utils

        if not hasattr(fa_utils, "is_flash_attn_available"):
            from transformers.utils import is_flash_attn_2_available

            fa_utils.is_flash_attn_available = is_flash_attn_2_available

        if not hasattr(fa_utils, "flash_attn_supports_top_left_mask"):
            from transformers.utils import (
                is_flash_attn_2_available,
                is_flash_attn_greater_or_equal_2_10,
            )

            def flash_attn_supports_top_left_mask() -> bool:
                if is_flash_attn_2_available():
                    return not is_flash_attn_greater_or_equal_2_10()
                return False

            fa_utils.flash_attn_supports_top_left_mask = (
                flash_attn_supports_top_left_mask
            )

        if not hasattr(tf_utils, "auto_docstring"):
            # Upstream: attaches a generated docstring; behaviorally inert.
            def auto_docstring(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]

                def decorator(obj):
                    return obj

                return decorator

            tf_utils.auto_docstring = auto_docstring

        if not hasattr(tf_utils, "can_return_tuple"):

            def can_return_tuple(func):
                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    return_dict = kwargs.pop("return_dict", None)
                    if return_dict is None:
                        return_dict = getattr(
                            getattr(self, "config", None), "use_return_dict", True
                        )
                    output = func(self, *args, **kwargs)
                    if not return_dict and not isinstance(output, tuple):
                        output = output.to_tuple()
                    return output

                return wrapper

            tf_utils.can_return_tuple = can_return_tuple

        if not hasattr(rope_utils, "dynamic_rope_update"):
            import torch

            def dynamic_rope_update(rope_forward):
                def longrope_frequency_update(self, position_ids, device):
                    seq_len = torch.max(position_ids) + 1
                    original_max = getattr(
                        self.config,
                        "original_max_position_embeddings",
                        self.config.max_position_embeddings,
                    )
                    if seq_len > original_max:
                        if not hasattr(self, "long_inv_freq"):
                            self.long_inv_freq, _ = self.rope_init_fn(
                                self.config, device, seq_len=original_max + 1
                            )
                        self.register_buffer(
                            "inv_freq", self.long_inv_freq, persistent=False
                        )
                    else:
                        self.original_inv_freq = self.original_inv_freq.to(device)
                        self.register_buffer(
                            "inv_freq", self.original_inv_freq, persistent=False
                        )

                def dynamic_frequency_update(self, position_ids, device):
                    seq_len = torch.max(position_ids) + 1
                    if seq_len > self.max_seq_len_cached:
                        inv_freq, self.attention_scaling = self.rope_init_fn(
                            self.config, device, seq_len=seq_len
                        )
                        self.register_buffer("inv_freq", inv_freq, persistent=False)
                        self.max_seq_len_cached = seq_len
                    if (
                        seq_len < self.original_max_seq_len
                        and self.max_seq_len_cached > self.original_max_seq_len
                    ):
                        self.original_inv_freq = self.original_inv_freq.to(device)
                        self.register_buffer(
                            "inv_freq", self.original_inv_freq, persistent=False
                        )
                        self.max_seq_len_cached = self.original_max_seq_len

                @functools.wraps(rope_forward)
                def wrapper(self, x, position_ids):
                    if "dynamic" in self.rope_type:
                        dynamic_frequency_update(self, position_ids, device=x.device)
                    elif self.rope_type == "longrope":
                        longrope_frequency_update(self, position_ids, device=x.device)
                    return rope_forward(self, x, position_ids)

                return wrapper

            rope_utils.dynamic_rope_update = dynamic_rope_update

        if "transformers.video_utils" not in sys.modules:
            try:
                import transformers.video_utils  # noqa: F401
            except ImportError:
                from transformers.image_utils import VideoInput

                video_utils = types.ModuleType("transformers.video_utils")
                video_utils.VideoInput = VideoInput
                sys.modules["transformers.video_utils"] = video_utils
                transformers.video_utils = video_utils

        if not hasattr(transformers, "AutoVideoProcessor"):
            # >=4.52 class the snapshot's processor declares for its
            # video_processor attribute. The KB pipeline never processes
            # video; resolve it to the image processor as a stand-in. The
            # metaclass makes ProcessorMixin's isinstance validation accept
            # whatever from_pretrained returns.
            class _AcceptsAnyInstance(type):
                def __instancecheck__(cls, instance):
                    return True

            class AutoVideoProcessor(metaclass=_AcceptsAnyInstance):
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    from transformers import AutoImageProcessor

                    return AutoImageProcessor.from_pretrained(*args, **kwargs)

            transformers.AutoVideoProcessor = AutoVideoProcessor
            # ProcessorMixin resolves attribute classes against its own
            # direct_transformers_import() module object, not
            # sys.modules["transformers"] — patch that lookup target too.
            import transformers.processing_utils as processing_utils

            if not hasattr(processing_utils.transformers_module, "AutoVideoProcessor"):
                processing_utils.transformers_module.AutoVideoProcessor = (
                    AutoVideoProcessor
                )

        from transformers.processing_utils import ProcessorMixin

        if not hasattr(ProcessorMixin, "_check_special_mm_tokens"):
            # >=4.51 validation helper: special multimodal token counts in
            # the raw text must survive tokenization untruncated.
            def _check_special_mm_tokens(self, text, text_inputs, modalities):
                for modality in modalities:
                    token_str = getattr(self, f"{modality}_token", None)
                    token_id = getattr(self, f"{modality}_token_id", None)
                    if token_str is None or token_id is None:
                        continue
                    ids_count = [
                        list(ids).count(token_id) for ids in text_inputs["input_ids"]
                    ]
                    text_count = [t.count(token_str) for t in text]
                    if ids_count != text_count:
                        raise ValueError(
                            f"Mismatch in `{modality}` token count between "
                            f"text and `input_ids` ({text_count} vs "
                            f"{ids_count}) — likely truncation dropped "
                            "special multimodal tokens."
                        )

            ProcessorMixin._check_special_mm_tokens = _check_special_mm_tokens

        try:
            from peft.tuners.lora import LoraLayer
        except ImportError:
            LoraLayer = None
        if LoraLayer is not None and not hasattr(LoraLayer, "_cast_input_dtype"):
            # peft >=0.15 helper the snapshot's custom_lora_module calls.
            def _cast_input_dtype(self, x, dtype):
                cast_enabled = getattr(self, "cast_input_dtype_enabled", True)
                if x is None:
                    return None
                if not cast_enabled or x.dtype == dtype:
                    return x
                return x.to(dtype=dtype)

            LoraLayer._cast_input_dtype = _cast_input_dtype

        # transformers >=4.51 tolerates plain-dict decoder sub-configs
        # (e.g. jina-v4's config.text_config) in
        # GenerationConfig.from_model_config; 4.50 calls .to_dict() on them
        # unconditionally. Retry on a copy with dict sub-configs wrapped.
        from transformers import PretrainedConfig
        from transformers.generation.configuration_utils import GenerationConfig

        if not getattr(GenerationConfig, "_marie_dict_subconfig_shim", False):
            import copy

            _orig_from_model_config = GenerationConfig.from_model_config.__func__

            @classmethod
            def _from_model_config(cls, model_config, *args, **kwargs):
                try:
                    return _orig_from_model_config(cls, model_config, *args, **kwargs)
                except AttributeError:
                    patched = copy.deepcopy(model_config)
                    for name in ("decoder", "generator", "text_config"):
                        sub = getattr(patched, name, None)
                        if isinstance(sub, dict):
                            setattr(patched, name, PretrainedConfig(**sub))
                    return _orig_from_model_config(cls, patched, *args, **kwargs)

            GenerationConfig.from_model_config = _from_model_config
            GenerationConfig._marie_dict_subconfig_shim = True

    @staticmethod
    def _materialize_sub_configs(config, model_path: str):
        """jina-v4's config subclasses the installed transformers' built-in
        Qwen2_5_VLConfig, which (pre-4.51) predates the text/vision
        sub-config split — checkpoint ``text_config``/``vision_config``
        dicts land on the config unconverted, while the snapshot's remote
        modeling code expects config objects. Convert them with the
        snapshot's own config classes."""
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        for name, cls_ref in (
            ("text_config", "qwen2_5_vl.Qwen2_5_VLTextConfig"),
            ("vision_config", "qwen2_5_vl.Qwen2_5_VLVisionConfig"),
        ):
            sub = getattr(config, name, None)
            if isinstance(sub, dict):
                sub_cls = get_class_from_dynamic_module(cls_ref, model_path)
                setattr(config, name, sub_cls(**sub))
        return config

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the jina-embeddings-v4 model."""
        try:
            from transformers import AutoConfig, AutoModel

            self._shim_transformers_compat()
            self.logger.info(f"Loading model from {model_path}")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config = self._materialize_sub_configs(config, model_path)
            model = AutoModel.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=self._torch_dtype,
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def get_embeddings(
        self,
        texts: List[str],
        truncation: bool = None,
        max_length: int = None,
        is_query: bool = False,
    ) -> EmbeddingsObject:
        """Generate embeddings for text content.

        Implements the EmbeddingsBase interface with additional support for
        query vs. passage distinction in retrieval tasks.

        Args:
            texts: List of text strings to embed.
            truncation: Whether to truncate long texts (default: True for jina-v4).
            max_length: Maximum sequence length.
            is_query: If True, use query prompt (shorter texts, search queries).
                     If False, use passage prompt (documents, longer texts).

        Returns:
            EmbeddingsObject containing embeddings and token count.
        """
        if not texts:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        # For retrieval task, use appropriate prompt
        prompt_name = "query" if is_query else "passage"

        with torch.no_grad():
            try:
                # jina-v4 uses encode_text method with task and prompt_name
                if hasattr(self.model, "encode_text"):
                    embeddings = self.model.encode_text(
                        texts=texts,
                        task=self.task,
                        prompt_name=prompt_name,
                        truncate_dim=self.truncate_dim,
                        batch_size=self.batch_size,
                    )
                else:
                    # Fallback for models without encode_text
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.batch_size,
                        truncation=truncation if truncation is not None else True,
                    )

                # Ensure numpy array. encode_text/encode_image return a
                # list of per-item tensors unless return_numpy is set.
                if isinstance(embeddings, list):
                    embeddings = torch.stack(
                        [
                            e if isinstance(e, torch.Tensor) else torch.as_tensor(e)
                            for e in embeddings
                        ]
                    )
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.float().cpu().numpy()

                # Truncate if needed
                if embeddings.shape[-1] > self.truncate_dim:
                    embeddings = embeddings[..., : self.truncate_dim]

                result = EmbeddingsObject()
                result.embeddings = embeddings
                result.total_tokens = -1  # Token count not readily available

                return result

            except Exception as e:
                self.logger.error(
                    f"Error during text embedding: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def get_image_embeddings(
        self,
        images: List[str],
    ) -> EmbeddingsObject:
        """Generate embeddings for images.

        Images are embedded into the same vector space as text,
        enabling direct text-to-image similarity search.

        Args:
            images: List of image paths or URLs.

        Returns:
            EmbeddingsObject containing image embeddings.
        """
        if not images:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        with torch.no_grad():
            try:
                # jina-v4 uses encode_image method
                if hasattr(self.model, "encode_image"):
                    embeddings = self.model.encode_image(
                        images=images,
                        task=self.task,
                        truncate_dim=self.truncate_dim,
                    )
                else:
                    raise NotImplementedError(
                        "Model does not support encode_image. "
                        "Ensure you are using jina-embeddings-v4."
                    )

                # Ensure numpy array. encode_text/encode_image return a
                # list of per-item tensors unless return_numpy is set.
                if isinstance(embeddings, list):
                    embeddings = torch.stack(
                        [
                            e if isinstance(e, torch.Tensor) else torch.as_tensor(e)
                            for e in embeddings
                        ]
                    )
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.float().cpu().numpy()

                # Truncate if needed
                if embeddings.shape[-1] > self.truncate_dim:
                    embeddings = embeddings[..., : self.truncate_dim]

                result = EmbeddingsObject()
                result.embeddings = embeddings
                result.total_tokens = 0

                return result

            except Exception as e:
                self.logger.error(
                    f"Error during image embedding: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def embed_text(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """Convenience method to embed text and return raw numpy array.

        Args:
            texts: List of text strings.
            is_query: Whether this is a query (vs. passage/document).

        Returns:
            Numpy array of embeddings, shape (len(texts), truncate_dim).
        """
        result = self.get_embeddings(texts, is_query=is_query)
        return result.embeddings

    def embed_images(
        self,
        images: List[str],
    ) -> np.ndarray:
        """Convenience method to embed images and return raw numpy array.

        Args:
            images: List of image paths or URLs.

        Returns:
            Numpy array of embeddings, shape (len(images), truncate_dim).
        """
        result = self.get_image_embeddings(images)
        return result.embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.truncate_dim
