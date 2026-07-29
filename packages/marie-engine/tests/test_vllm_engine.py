import logging
from unittest.mock import Mock

from marie.engine.vllm_engine import VLLMEngine


def test_batch_generate_logs_traceback(caplog) -> None:
    engine = object.__new__(VLLMEngine)
    engine.is_multimodal = False
    engine.system_prompt = "system"
    engine.logger = logging.getLogger("VLLMEngine")
    engine.llm = Mock()
    engine.llm.generate.side_effect = RuntimeError("engine core failed")
    engine._generate_prompt = lambda *_args: "prompt"
    engine._get_structured_outputs_params = lambda *_args: None

    with caplog.at_level(logging.ERROR, logger="VLLMEngine"):
        result = engine.batch_generate(["content"])

    assert result == ["ERROR: Batch Inference failed"]
    assert caplog.records[-1].exc_info is not None
    assert caplog.records[-1].exc_info[0] is RuntimeError
    assert "Traceback (most recent call last)" in caplog.text
    assert "RuntimeError: engine core failed" in caplog.text
