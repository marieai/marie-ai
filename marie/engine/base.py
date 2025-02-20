import hashlib
import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, List, Union

import diskcache as dc
import platformdirs
from PIL import Image

from marie.engine.guided import GuidedMode


def cached(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        if self.cache is False:
            return func(self, *args, **kwargs)

        # get string representation from args and kwargs
        key = hash(str(args) + str(kwargs))
        key = hashlib.sha256(f"{key}".encode()).hexdigest()

        if key in self.cache:
            return self.cache[key]

        result = func(self, *args, **kwargs)
        self.cache[key] = result
        return result

    return wrapper


class EngineLM(ABC):
    """
    Abstract base class for Language Model (LM) engines.
    Supports both text-based and multimodal inference.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and intelligent assistant."

    def __init__(
        self,
        model_string: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        cache: Union[dc.Cache, bool] = False,
    ):
        """
        Base class for the engines.

        :param model_string: The model string to use.
        :param system_prompt: The system prompt to use. Defaults to "You are a helpful, creative, and smart assistant."
        :param is_multimodal: Whether the model is multimodal. Defaults to False.
        :param cache: The cache to use. Defaults to False. Note that cache can also be a diskcache.Cache object.
        """

        root = platformdirs.user_cache_dir("marie")
        default_cache_path = os.path.join(root, f"cache_model_{model_string}.db")

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        # cache resolution
        if isinstance(cache, dc.Cache):
            self.cache = cache
        elif cache is True:
            self.cache = dc.Cache(default_cache_path)
        elif cache is False:
            self.cache = False
        else:
            raise ValueError(
                "Cache argument must be a diskcache.Cache object or a boolean."
            )

    @abstractmethod
    def _generate_from_multiple_input(
        self,
        prompt: Union[
            List[Union[Image.Image, bytes, str]],  # Single multimodal input
            List[List[Union[Image.Image, bytes, str]]],  # Batch multimodal inputs
        ],
        system_prompt: str = None,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def _generate_from_single_prompt(
        self,
        prompt: Union[str, List[str]],
        system_prompt: str = None,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        pass

    def generate(
        self,
        content: Union[
            str,  # Single text prompt
            List[str],  # Batch text prompts
            List[Union[Image.Image, bytes, str]],  # Single multimodal input
            List[List[Union[Image.Image, bytes, str]]],  # Batch multimodal inputs
        ],
        system_prompt: Union[str | List[Union[str, bytes]]] = None,
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
        **kwargs,
    ):
        """
        Handles both single and batch inference for text and multimodal inputs.

        :param content: The input prompt(s), which can be text, images, or multimodal inputs.
        :param system_prompt: Optional system-level instructions.
        :param guided_mode: Optional guided mode.
        :param guided_params: Optional guided parameters for the guided mode.
        :param kwargs: Additional parameters for generation.
        :return: The generated response(s).
        """

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        # Single or Batch Text Input  (e.g., "prompt" or ["prompt1", "prompt2"])
        if isinstance(content, str) or (
            isinstance(content, list) and all(isinstance(item, str) for item in content)
        ):
            return self._generate_from_single_prompt(
                content=content,
                system_prompt=sys_prompt_arg,
                guided_mode=guided_mode,
                guided_params=guided_params,
                **kwargs,
            )

        # Multimodal Inputs
        is_multimodal_single = (
            isinstance(content, list)
            and any(
                isinstance(item, (Image.Image, bytes)) for item in content
            )  # At least one image
            and any(
                isinstance(item, str) for item in content
            )  # At least one text input
            and not any(isinstance(sublist, list) for sublist in content)  # Not a batch
        )

        is_multimodal_batch = isinstance(content, list) and all(
            isinstance(sublist, list)
            and any(
                isinstance(el, (Image.Image, bytes)) for el in sublist
            )  # Each sublist has at least one image
            and any(
                isinstance(el, str) for el in sublist
            )  # Each sublist has at least one text input
            for sublist in content
        )

        if (is_multimodal_single or is_multimodal_batch) and not self.is_multimodal:
            raise NotImplementedError(
                "Multimodal generation flag is not set, but multimodal input is provided. "
                "Is this model multimodal?"
            )

        return self._generate_from_multiple_input(
            content=content,
            system_prompt=sys_prompt_arg,
            guided_mode=guided_mode,
            guided_params=guided_params,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        pass
