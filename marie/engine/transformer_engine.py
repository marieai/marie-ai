import os.path
import time
from typing import Dict, List, Union

import diskcache as dc
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from marie.engine.base import EngineLM, cached
from marie.engine.engine_utils import (
    as_bytes,
    open_ai_like_formatting,
    process_vision_info,
)
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings


class TransformerEngine(EngineLM):
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        model_name_or_path: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = True,
        cache: Union[dc.Cache, bool] = False,
        processor_kwargs: Dict = None,
    ):
        super().__init__(
            model_string=model_name_or_path,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )

        self.logger = MarieLogger(self.__class__.__name__).logger
        processor_kwargs = processor_kwargs or {}
        devices = ['cuda'] if torch.cuda.is_available() else ['cpu']

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=True, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]
        self.device_str = self.device.type

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",  # Let the library handle device allocation automatically
            quantization_config=nf4_config if self.device.type == "cuda" else None,
        )  # .to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, **processor_kwargs
        )

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
        self,
        content: str,
        system_prompt: str = None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        return self.hf_generate(content, system_prompt)

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes | Image.Image]],
        system_prompt=None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        formatted_content = open_ai_like_formatting(content)

        return self.hf_generate(formatted_content, system_prompt)

    def __call__(self, content, **kwargs):
        return self.generate(content, **kwargs)

    @torch.inference_mode()
    def hf_generate(self, content, system_prompt=None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        self.logger.info(f"Raw text: {messages}")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.logger.info(f"Generated text: {text}")

        image_inputs, _ = process_vision_info(messages) or ([], [])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=_,
            padding=True,
            return_tensors="pt",
        )  # .to(self.device)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        start_time = time.time()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,  # output length
            num_beams=1,  # Use beam search
            do_sample=False,  # Disable sampling
            repetition_penalty=1.2,  # Penalize repeated sequences
            top_p=None,  # Disable nucleus sampling
            temperature=None,  # Disable temperature sampling
            top_k=None,  # Disable top-k sampling
            return_dict_in_generate=True,
            output_scores=True,
        )

        end_time = time.time()
        total_time = end_time - start_time
        generated_ids = generated_ids["sequences"]  # sequences, scores
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        num_tokens = sum(len(ids) for ids in generated_ids_trimmed)
        tokens_per_second = num_tokens / total_time if total_time > 0 else 0

        print(f"Generated {num_tokens} tokens in {total_time:.2f} seconds.")
        print(f"Token Processing Speed: {tokens_per_second:.2f} tokens/sec.")
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text


if __name__ == "__main__":
    # Qwen/Qwen2-VL-7B-Instruct
    # Qwen/Qwen2-VL-2B-Instruct

    # Qwen/Qwen2.5-VL-7B-Instruct
    # Qwen/Qwen2.5-VL-3B-Instruct

    # # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/discussions/10
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28

    engine = TransformerEngine(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        system_prompt="You are a helpful assistant for processing EOB documents.",
        is_multimodal=True,
        cache=False,
        processor_kwargs={  # All parameters will be passed dynamically to the processor
            'min_pixels': 256 * 28 * 28,
            'max_pixels': 1280 * 28 * 28,  # 800, 1000, 1280
        },
    )

    image_path = os.path.expanduser(
        "~/datasets/corr-indexer/testdeck-raw-01/images/corr-indexing/test/152658564_4.png"
    )
    image = as_bytes(image_path)
    image = Image.open(image_path).convert("RGB")

    prompt = """
    Extract only key-value pairs from the given image. Identify and associate relevant fields with their corresponding values while maintaining accuracy and structure.

    ### Follow these rules strictly:
    1. **Extract only structured key-value pairs. Do not generate explanations, summaries, or extra text.**
    2. **Preserve key names exactly as they appear in the image.**
    3. **If a key has no corresponding value, return `Key: [MISSING]`.**
    4. **If no key-value pairs are found, return exactly: "No key-value pairs found."**

    ### Output Format (Strictly Follow This):
    Key1: Value1; Key2: Value2; Key3: Value3; ...

    Your response should contain nothing except the extracted key-value pairs in the format specified above. Avoid adding any interpretations, explanations, or greetings.
    """

    print(prompt)

    result = engine([image, prompt], system_prompt="You are a helpful assistant.")
    print(result)
