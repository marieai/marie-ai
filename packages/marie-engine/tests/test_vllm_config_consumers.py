from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_constructor_uses_external_settings_and_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("""schema_version: 1
defaults:
  vllm:
    loading: {dtype: bfloat16, quantization: null, load_format: auto}
    runtime: {max_model_len: 2048, gpu_memory_utilization: 0.6}
    processor: {min_pixels: 784, fps: 1}
    generation: {temperature: 0, max_tokens: 64}
    prompt: {system: configured system, template: 'SYSTEM_PROMPT_PLACEHOLDER / QUESTION_PLACEHOLDER'}
models:
  new_alias: {name: acme/new-model, multimodal: true}
""")
    monkeypatch.setenv("MARIE_MODEL_CATALOG", str(path))
    import marie.engine.vllm_config as adapter
    import marie.engine.vllm_engine as module
    from marie.registry.model_config import ModelMetadata

    monkeypatch.setattr(
        module.ModelRegistry,
        "metadata",
        lambda *args, **kwargs: ModelMetadata(config={"vision_config": {}}),
    )

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tokenizer = SimpleNamespace(name="provider-tokenizer")

        def get_tokenizer(self):
            return self.tokenizer

    monkeypatch.setattr(adapter, "LLM", LLM)
    engine = module.VLLMEngine(
        "new_alias",
        processor_kwargs={"fps": 2},
        runtime_kwargs={"max_model_len": 1024},
        max_tokens=32,
        system_prompt="caller system",
    )
    assert engine.llm.kwargs["model"] == "acme/new-model"
    assert engine.llm.kwargs["max_model_len"] == 1024
    assert engine.llm.kwargs["mm_processor_kwargs"] == {"min_pixels": 784, "fps": 2}
    assert engine.generation_defaults["max_tokens"] == 32
    assert engine.system_prompt == "caller system"
    assert engine.prompt == "SYSTEM_PROMPT_PLACEHOLDER / QUESTION_PLACEHOLDER"
    assert engine.is_multimodal is True
    assert engine.tokenizer is engine.llm.tokenizer


def test_sampling_uses_configured_defaults_and_request_overrides() -> None:
    from marie.engine.vllm_engine import VLLMEngine

    engine = object.__new__(VLLMEngine)
    engine.is_multimodal = False
    engine.system_prompt = "system"
    engine.logger = logging.getLogger("configured-vllm-test")
    engine.generation_defaults = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 10,
        "max_tokens": 256,
        "n": 1,
        "stop_token_ids": [7],
        "structured_outputs": {"disable_any_whitespace": False},
    }
    engine.tokenizer = SimpleNamespace(tokenize=lambda text: text.split())
    engine._generate_prompt = lambda messages, system: "prompt"
    engine._get_structured_outputs_params = lambda *args: None

    class LLM:
        def generate(self, inputs, *, sampling_params):
            self.sampling_params = sampling_params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")])]

    engine.llm = LLM()
    assert engine.batch_generate(["text"], temperature=0, max_tokens=32) == ["answer"]
    params = engine.llm.sampling_params
    assert params.temperature == 0
    assert params.max_tokens == 32
    assert params.top_p == 1
    assert params.stop_token_ids == [7]
    engine.batch_generate(["text"])
    assert engine.llm.sampling_params.temperature == 0.7
    assert engine.llm.sampling_params.top_k == 10
    assert engine.llm.sampling_params.top_p == 0.9


def test_structured_output_options_are_configurable() -> None:
    from marie.engine.vllm_engine import VLLMEngine

    engine = object.__new__(VLLMEngine)
    engine.generation_defaults = {
        "structured_outputs": {"disable_any_whitespace": False}
    }
    params = engine._get_structured_outputs_params(guided_json={"type": "object"})
    assert params.disable_any_whitespace is False
    params = engine._get_structured_outputs_params(
        guided_json={"type": "object"},
        guided_backend="xgrammar:disable-any-whitespace",
    )
    assert params.disable_any_whitespace is True


def test_explicit_image_settings_skip_helper_defaults() -> None:
    from marie.engine.engine_utils import open_ai_like_formatting
    from PIL import Image

    with Image.new("RGB", (28, 28)) as frame:
        message = open_ai_like_formatting(
            [frame, "text"], use_defaults=False, min_pixels=64
        )[0]
        assert message["min_pixels"] == 64
        assert "max_pixels" not in message
        assert "use_defaults" not in message


def test_null_structured_output_defaults_allow_request_schema() -> None:
    from marie.engine.vllm_engine import VLLMEngine

    engine = object.__new__(VLLMEngine)
    engine.generation_defaults = {"structured_outputs": None}
    assert engine._get_structured_outputs_params() is None
    params = engine._get_structured_outputs_params(guided_json={"type": "object"})
    assert params.json == {"type": "object"}


def test_configured_prompt_template_is_used() -> None:
    from marie.engine.vllm_engine import VLLMEngine

    engine = object.__new__(VLLMEngine)
    engine.prompt = "system=SYSTEM_PROMPT_PLACEHOLDER; question=QUESTION_PLACEHOLDER"
    messages = [{"role": "user", "content": [{"type": "text", "text": "question"}]}]
    assert (
        engine._generate_prompt(messages, "configured system")
        == "system=configured system; question=question"
    )


def test_image_settings_and_request_processor_options_reach_vllm() -> None:
    from marie.engine.vllm_engine import VLLMEngine
    from PIL import Image

    engine = object.__new__(VLLMEngine)
    engine.is_multimodal = True
    engine.system_prompt = "system"
    engine.logger = logging.getLogger("image-config-test")
    engine.generation_defaults = {"max_tokens": 32}
    engine.default_mm_processor_kwargs = {"min_pixels": 784, "max_pixels": 784}
    engine.model_config = SimpleNamespace(processor={"num_crops": 16})
    engine.tokenizer = SimpleNamespace(tokenize=lambda text: text.split())
    engine._generate_prompt = lambda *args: "prompt"

    class LLM:
        def generate(self, inputs, *, sampling_params):
            self.inputs = inputs
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")])]

    engine.llm = LLM()
    with Image.new("RGB", (280, 280)) as image:
        assert engine.batch_generate(
            [[image, "question"]], mm_processor_kwargs={"num_crops": 2}
        ) == ["answer"]
    request = engine.llm.inputs[0]
    assert request["multi_modal_data"]["image"][0].size == (28, 28)
    assert request["mm_processor_kwargs"] == {"num_crops": 2}
    assert engine.model_config.processor == {"num_crops": 16}


@pytest.mark.parametrize("size", [(1000, 1), (1, 1000), (100, 100)])
def test_configured_pixel_bounds_survive_dimension_rounding(size) -> None:
    from marie.engine.engine_utils import resize_image
    from PIL import Image

    with Image.new("RGB", size) as image:
        resized = resize_image(image, max_pixels=200)
        assert resized.width * resized.height <= 200
        enlarged = resize_image(image, min_pixels=12001)
        assert enlarged.width * enlarged.height >= 12001


@pytest.mark.parametrize(
    "generation",
    [
        "{temperature: -1}",
        "{typo_max_tokens: 32}",
        "{structured_outputs: {unsupported_option: true}}",
    ],
)
def test_invalid_generation_fails_before_loading_weights(
    tmp_path, monkeypatch, generation
) -> None:
    import marie.engine.vllm_engine as module
    from marie.registry.model_config import ModelMetadata

    path = tmp_path / "models.yaml"
    path.write_text(
        f"schema_version: 1\ndefaults:\n  vllm:\n    generation: {generation}\nmodels:\n  custom: {{name: acme/custom, multimodal: true}}\n"
    )
    monkeypatch.setenv("MARIE_MODEL_CATALOG", str(path))
    monkeypatch.setattr(
        module.ModelRegistry,
        "metadata",
        lambda *a, **kw: ModelMetadata(config={"vision_config": {}}),
    )
    monkeypatch.setattr(
        module,
        "create_llm_instance",
        lambda config: pytest.fail("Invalid settings must fail before loading weights"),
    )
    with pytest.raises(ValueError, match=str(path)):
        module.VLLMEngine("custom")
