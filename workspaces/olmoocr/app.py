import base64
import io
import json

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from marie.utils.docs import frames_from_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from olmocr.prompts.prompts import (
#     build_finetuning_prompt,
# )

# Load the OCR model and processor from Hugging Face
try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
    ).eval()
    model.to(device)
except ImportError as e:
    processor = None
    model = None
    print(f"Error loading model: {str(e)}. Please ensure PyTorch is installed.")
except ValueError as e:
    processor = None
    model = None
    print(f"Error with model configuration: {str(e)}")


# This is a base prompt that will be used for training and running the fine tuned model
# It's simplified from the prompt which was used to generate the silver data, and can change from dataset to dataset
def build_finetuning_prompt(base_text: str) -> str:
    return (
        f"Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. "
        f"Just return the JSON representation of this document as if you were reading it naturally.\n"
        f"""
          Detect all table headers in the image and return their locations in the form of coordinates. 
        
        - **Expected Table Header Columns:**  
          1. SERVICE DATES  
          2. PL  
          3. SERVICE CODES  
          4. NUM SVC  
          5. SUBMITTED CHARGES  
          6. NEGOTIATED AMOUNT  
          7. COPAY AMOUNT  
          8. NOT PAYABLE  
          9. SEE REMARKS  
          10. DEDUCTIBLE  
          11. CO INSURANCE  
          12. PATIENT RESP  
          13. PAYABLE AMOUNT                   
        """
        f"Do not hallucinate.\n"
        f"Output QwenVL HTML\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


def process_image(pdf_file, anchor_text):
    """
    Process the uploaded PDF file one page at a time, yielding HTML for each page
    with its image and extracted text.
    """
    if processor is None or model is None:
        yield "<p>Error: Model could not be loaded. Check environment setup (PyTorch may be missing) or model compatibility.</p>"
        return

    # Check if a PDF file was uploaded
    if pdf_file is None:
        yield "<p>Please upload a file.</p>"
        return

    try:
        pages = frames_from_file(pdf_file)
        pages = [Image.fromarray(page) for page in pages]
    except Exception as e:
        yield f"<p>Error converting PDF to images: {str(e)}</p>"
        return

    html = '<div><button onclick="copyAll()" style="margin-bottom: 10px;">Copy All</button></div><div id="pages">'
    yield html  # Start with the header

    # Process each page incrementally
    for i, page in enumerate(pages):
        # Convert the page image to base64 for embedding in HTML
        buffered = io.BytesIO()
        page.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_data = f"data:image/png;base64,{image_base64}"
        # Build the prompt, using document metadata
        prompt = build_finetuning_prompt(anchor_text)

        print("prompt", prompt)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            main_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}

            # Generate the output
            output = model.generate(
                **inputs,
                temperature=0.7,
                max_new_tokens=2048,
                num_return_sequences=1,
                do_sample=True,
            )

            print(output)
            # Decode the output
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text = processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
            print(text)
            natural_text = json.loads(text)["natural_text"]
            text = natural_text
        except Exception as e:
            text = f"Error extracting text: {str(e)}"

        # Generate HTML for this page's section
        textarea_id = f"text{i + 1}"
        page_html = f'''
        <div class="page" style="margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 20px;">
            <h3>Page {i + 1}</h3>
            <div style="display: flex; align-items: flex-start;">
                <img src="{img_data}" alt="Page {i + 1}" style="max-width: 300px; margin-right: 20px;">
                <div style="flex-grow: 1;">
                    <textarea id="{textarea_id}" rows="10" style="width: 100%;">{text}</textarea>
                    <button onclick="copyText('{textarea_id}')" style="margin-top: 5px;">Copy</button>
                </div>
            </div>
        </div>
        '''

        # Append this page to the existing HTML and yield the updated content
        html += page_html
        yield html

    # After all pages are processed, close the div and add JavaScript
    html += '</div>'
    html += '''
    <script>
    function copyText(id) {
        var text = document.getElementById(id);
        text.select();
        document.execCommand("copy");
    }
    function copyAll() {
        var texts = document.querySelectorAll("#pages textarea");
        var allText = Array.from(texts).map(t => t.value).join("\\n\\n");
        navigator.clipboard.writeText(allText);
    }
    </script>
    '''
    yield html  # Final yield with complete content and scripts


with gr.Blocks(title=" Text Extractor") as demo:
    gr.Markdown("#  Text Extractor")
    gr.Markdown(
        "Upload a file and click 'Extract Text' to see each page's image and extracted text incrementally."
    )
    with gr.Row():
        image_input = gr.File(
            label="Upload Image", file_types=[".png", ".jpg", ".jpeg", ".pdf"]
        )
        textarea_input = gr.TextArea(
            label="Additional Text Input (Optional)",
            lines=3,
            placeholder="Enter additional text here...",
        )

        submit_btn = gr.Button("Extract Text")
    output_html = gr.HTML()
    submit_btn.click(
        fn=process_image, inputs=[image_input, textarea_input], outputs=output_html
    )

demo.launch()

# https://github.com/17794/olmocr_util/blob/9d5ad2d788a02d27985fba98be3f29597c1aa1ce/pdf_processor.py
# https://github.com/allenai/olmocr/tree/main/tests
# https://github.com/allenai/olmocr/blob/main/olmocr/viewer/dolmaviewer.py
