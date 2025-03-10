import base64
import io
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    use_safetensors=True,
)
model = model.eval().cuda()

UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def run_GOT(image, got_mode, fine_grained_mode="", ocr_color="", ocr_box=""):
    unique_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
    result_path = os.path.join(RESULTS_FOLDER, f"{unique_id}.html")

    shutil.copy(image, image_path)

    try:
        if got_mode == "plain texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='ocr')
            return res, None
        elif got_mode == "format texts OCR":
            res = model.chat(
                tokenizer,
                image_path,
                ocr_type='format',
                render=True,
                save_render_file=result_path,
            )
        elif got_mode == "plain multi-crop OCR":
            res = model.chat_crop(tokenizer, image_path, ocr_type='ocr')
            return res, None
        elif got_mode == "format multi-crop OCR":
            res = model.chat_crop(
                tokenizer,
                image_path,
                ocr_type='format',
                render=True,
                save_render_file=result_path,
            )
        elif got_mode == "plain fine-grained OCR":
            res = model.chat(
                tokenizer,
                image_path,
                ocr_type='ocr',
                ocr_box=ocr_box,
                ocr_color=ocr_color,
            )
            return res, None
        elif got_mode == "format fine-grained OCR":
            res = model.chat(
                tokenizer,
                image_path,
                ocr_type='format',
                ocr_box=ocr_box,
                ocr_color=ocr_color,
                render=True,
                save_render_file=result_path,
            )

        # res_markdown = f"$$ {res} $$"
        res_markdown = res

        if "format" in got_mode and os.path.exists(result_path):
            with open(result_path, 'r') as f:
                html_content = f.read()
            encoded_html = base64.b64encode(html_content.encode('utf-8')).decode(
                'utf-8'
            )
            iframe_src = f"data:text/html;base64,{encoded_html}"
            iframe = f'<iframe src="{iframe_src}" width="100%" height="600px"></iframe>'
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">Download Full Result</a>'
            return res_markdown, f"{download_link}<br>{iframe}"
        else:
            return res_markdown, None
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


def task_update(task):
    if "fine-grained" in task:
        return [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ]
    else:
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ]


def fine_grained_update(task):
    if task == "box":
        return [
            gr.update(visible=False, value=""),
            gr.update(visible=True),
        ]
    elif task == 'color':
        return [
            gr.update(visible=True),
            gr.update(visible=False, value=""),
        ]


def cleanup_old_files():
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for file_path in Path(folder).glob('*'):
            if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()


title_html = """
<h2> <span class="gradient-text" id="text">General OCR Theory</span><span class="plain-text">: Towards OCR-2.0 via a Unified End-to-end Model</span></h2>
<a href="https://huggingface.co/ucaslcl/GOT-OCR2_0">[😊 Hugging Face]</a> 
<a href="https://arxiv.org/abs/2409.01704">[📜 Paper]</a> 
<a href="https://github.com/Ucas-HaoranWei/GOT-OCR2.0/">[🌟 GitHub]</a> 
"""

with gr.Blocks() as demo:
    gr.HTML(title_html)
    gr.Markdown(
        """
    "🔥🔥🔥This is the official online demo of GOT-OCR-2.0 model!!!"

    ### Demo Guidelines
    You need to upload your image below and choose one mode of GOT, then click "Submit" to run GOT model. More characters will result in longer wait times.
    - **plain texts OCR & format texts OCR**: The two modes are for the image-level OCR.
    - **plain multi-crop OCR & format multi-crop OCR**: For images with more complex content, you can achieve higher-quality results with these modes.
    - **plain fine-grained OCR & format fine-grained OCR**: In these modes, you can specify fine-grained regions on the input image for more flexible OCR. Fine-grained regions can be coordinates of the box, red color, blue color, or green color.
    """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="upload your image")
            task_dropdown = gr.Dropdown(
                choices=[
                    "plain texts OCR",
                    "format texts OCR",
                    "plain multi-crop OCR",
                    "format multi-crop OCR",
                    "plain fine-grained OCR",
                    "format fine-grained OCR",
                ],
                label="Choose one mode of GOT",
                value="plain texts OCR",
            )
            fine_grained_dropdown = gr.Dropdown(
                choices=["box", "color"], label="fine-grained type", visible=False
            )
            color_dropdown = gr.Dropdown(
                choices=["red", "green", "blue"], label="color list", visible=False
            )
            box_input = gr.Textbox(
                label="input box: [x1,y1,x2,y2]",
                placeholder="e.g., [0,0,100,100]",
                visible=False,
            )
            submit_button = gr.Button("Submit")

        with gr.Column():
            ocr_result = gr.Textbox(label="GOT output")

    with gr.Column():
        gr.Markdown(
            "**If you choose the mode with format, the mathpix result will be automatically rendered as follows:**"
        )
        html_result = gr.HTML(label="rendered html", show_label=True)

    gr.Examples(
        examples=[
            ["assets/coco.jpg", "plain texts OCR", "", "", ""],
            ["assets/en_30.png", "plain texts OCR", "", "", ""],
            ["assets/table.jpg", "format texts OCR", "", "", ""],
            ["assets/eq.jpg", "format texts OCR", "", "", ""],
            ["assets/exam.jpg", "format texts OCR", "", "", ""],
            ["assets/giga.jpg", "format multi-crop OCR", "", "", ""],
            [
                "assets/aff2.png",
                "plain fine-grained OCR",
                "box",
                "",
                "[409,763,756,891]",
            ],
            ["assets/color.png", "plain fine-grained OCR", "color", "red", ""],
        ],
        inputs=[
            image_input,
            task_dropdown,
            fine_grained_dropdown,
            color_dropdown,
            box_input,
        ],
        outputs=[ocr_result, html_result],
        fn=run_GOT,
        label="examples",
    )

    task_dropdown.change(
        task_update,
        inputs=[task_dropdown],
        outputs=[fine_grained_dropdown, color_dropdown, box_input],
    )
    fine_grained_dropdown.change(
        fine_grained_update,
        inputs=[fine_grained_dropdown],
        outputs=[color_dropdown, box_input],
    )

    submit_button.click(
        run_GOT,
        inputs=[
            image_input,
            task_dropdown,
            fine_grained_dropdown,
            color_dropdown,
            box_input,
        ],
        outputs=[ocr_result, html_result],
    )

if __name__ == "__main__":
    cleanup_old_files()
    demo.launch()
