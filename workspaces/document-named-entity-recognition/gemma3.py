import os

import gradio as gr
import torch
from transformers import pipeline

hf_token = None

# Load the Gemma 3 pipeline
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
    use_auth_token=hf_token,
)


def get_response(message, chat_history, image):
    # Check if image is provided
    if image is None:
        chat_history.append((message, "Please upload an image (required)"))
        return "", chat_history

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        }
    ]

    user_content = [{"type": "image", "image": image}]

    # Add text message if provided
    if message:
        user_content.append({"type": "text", "text": message})

    messages.append({"role": "user", "content": user_content})

    # Call the pipeline
    output = pipe(text=messages, max_new_tokens=1024)

    try:
        response = output[0]["generated_text"][-1]["content"]
        chat_history.append((message, response))
    except (KeyError, IndexError, TypeError) as e:
        error_message = f"Error processing the response: {str(e)}"
        chat_history.append((message, error_message))

    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# Gemma 3 Image Chat")
    gr.Markdown(
        "Chat with Gemma 3 about images. Image upload is required for each message."
    )

    chatbot = gr.Chatbot()

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type your message here about the image...",
            scale=4,
        )
        img = gr.Image(type="pil", label="Upload image (required)", scale=1)

    submit_btn = gr.Button("Send")

    # Clear button to reset the interface
    clear_btn = gr.Button("Clear")

    def clear_interface():
        return "", [], None

    submit_btn.click(get_response, inputs=[msg, chatbot, img], outputs=[msg, chatbot])

    msg.submit(get_response, inputs=[msg, chatbot, img], outputs=[msg, chatbot])

    clear_btn.click(clear_interface, inputs=None, outputs=[msg, chatbot, img])

if __name__ == "__main__":
    demo.launch()
