import gradio as gr
from PIL import Image

from marie.executor import NerExtractionExecutor
from marie.utils.image_utils import hash_file

executor = NerExtractionExecutor("rms/layoutlmv3-large-20221118-001-best")


def process_image(image):

    width, height = image.size

    img_path = "/tmp/gradio.png"
    image.save(img_path)

    checksum = hash_file(img_path)
    docs = None
    kwa = {"checksum": checksum, "img_path": img_path}
    results = executor.extract(docs, **kwa)
    print(results)

    ner_path = f"/tmp/tensors/ner_{checksum}_0.png"
    prediction_path = f"/tmp/tensors/prediction_{checksum}_0.png"

    ner_image = Image.open(ner_path).convert("RGB")
    pred_image = Image.open(prediction_path).convert("RGB")

    return results, ner_image, pred_image


def interface():
    title = "Extracting Named Entity Recognition / Key Value pair extraction"
    description = """<p>This particular model is fine-tuned from Correspondence Indexing Dataset on LayoutLMv3-Large</p>"""

    article = (
        "<p style='text-align: center'><a href='https://arxiv.org/abs/2204.08387'"
        " target='_blank'>LayoutLMv3: Multi-modal Pre-training for Visually-Rich"
        " Document Understanding</a> </p>"
    )
    # examples = [['sample-01.png']]
    examples = []

    iface = gr.Interface(
        fn=process_image,
        inputs=[
            gr.inputs.Image(type="pil"),
        ],
        outputs=[
            gr.outputs.JSON(),
            gr.outputs.Image(type="pil", label="annotated image"),
            gr.outputs.Image(type="pil", label="predictions"),
        ],
        title=title,
        description=description,
        article=article,
        examples=examples,
        theme="default",
        css=".footer{display:none !important}",
        live=False,
    )

    iface.launch(debug=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    interface()
