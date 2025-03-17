import subprocess

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_optim='cpu',  # Optional: can specify more configuration options like bnb_optim or other tuning params
)


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit loading
    llm_int8_threshold=6.0,  # Threshold for mixed-precision
    llm_int8_skip_modules=["lm_head"],  # Skip specific modules
)


models = {
    # "microsoft/Phi-3.5-vision-instruct":
    #   AutoModelForCausalLM.from_pretrained(
    #       "microsoft/Phi-3.5-vision-instruct",
    #       trust_remote_code=True,
    #       torch_dtype="auto",
    #       _attn_implementation="flash_attention_2",
    #       quantization_config=quantization_config
    #       )
    #       .cuda()
}

processors = {
    # "microsoft/Phi-3.5-vision-instruct": AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
}


model_id = "microsoft/Phi-3.5-vision-instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    attn_implementation='flash_attention_2',
    quantization_config=bnb_config,
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    #   num_crops=16
)


DESCRIPTION = (
    "[Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)"
)

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"


def run_example(image, text_input=None, model_id="microsoft/Phi-3.5-vision-instruct"):
    # model = models[model_id]
    # processor = processors[model_id]

    prompt = f"{user_prompt}<|image_1|>\n{text_input}{prompt_suffix}{assistant_prompt}"
    image = Image.fromarray(image).convert("RGB")

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Phi-3.5 Input"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                text_input = gr.Textbox(label="Question")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")

        submit_btn.click(run_example, [input_img, text_input], [output_text])

demo.queue(api_open=False)
demo.launch(debug=True, show_api=False)
