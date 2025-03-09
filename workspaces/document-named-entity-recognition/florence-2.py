import subprocess

import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"},shell=True)

# HuggingFaceM4/Florence-2-DocVQA
# MODEL_ID = "HuggingFaceM4/Florence-2-DocVQA"
MODEL_ID = "microsoft/Florence-2-large"
model = (
    AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    .to("cuda")
    .eval()
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

TITLE = "# [Florence-2-Doc]"
DESCRIPTION = "The demo for Florence-2 fine-tuned "

colormap = [
    'blue',
    'orange',
    'green',
    'purple',
    'brown',
    'pink',
    'gray',
    'olive',
    'cyan',
    'red',
    'lime',
    'indigo',
    'violet',
    'aqua',
    'magenta',
    'coral',
    'gold',
    'tan',
    'skyblue',
]


def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer


def process_image(image, text_input=None):
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    task_prompt = '<DocVQA>'
    results = run_example(task_prompt, image, text_input)[task_prompt].replace(
        "<pad>", ""
    )
    return results


css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                text_input = gr.Textbox(label="Text Input (optional)")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")

        # gr.Examples(
        #     examples=[
        #         ["hunt.jpg", 'What is this image?'],
        #         ["idefics2_architecture.png", 'How many tokens per image does it use?'],
        #         ["idefics2_architecture.png", "What type of encoder does the model use?"],
        #         ["image.jpg", "What's the share of Industry Switchers Gained?"]
        #     ],
        #     inputs=[input_img, text_input],
        #     outputs=[output_text],
        #     fn=process_image,
        #     cache_examples=True,
        #     label='Try the examples below'
        # )

        submit_btn.click(process_image, [input_img, text_input], [output_text])

demo.launch(debug=True)
