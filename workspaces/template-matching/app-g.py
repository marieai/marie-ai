import gradio as gr
import torch as torch

use_cuda = torch.cuda.is_available()

# https://github.com/gradio-app/gradio/issues/2316
ROI_coordinates = {
    'x_temp': 0,
    'y_temp': 0,
    'x_new': 0,
    'y_new': 0,
    'clicks': 0,
}


def get_select_coordinates(img, evt: gr.SelectData):
    sections = []
    # update new coordinates
    ROI_coordinates['clicks'] += 1
    ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['y_temp'] = ROI_coordinates['y_new']
    ROI_coordinates['x_new'] = evt.index[0]
    ROI_coordinates['y_new'] = evt.index[1]
    # compare start end coordinates
    x_start = (
        ROI_coordinates['x_new']
        if (ROI_coordinates['x_new'] < ROI_coordinates['x_temp'])
        else ROI_coordinates['x_temp']
    )
    y_start = (
        ROI_coordinates['y_new']
        if (ROI_coordinates['y_new'] < ROI_coordinates['y_temp'])
        else ROI_coordinates['y_temp']
    )
    x_end = (
        ROI_coordinates['x_new']
        if (ROI_coordinates['x_new'] > ROI_coordinates['x_temp'])
        else ROI_coordinates['x_temp']
    )
    y_end = (
        ROI_coordinates['y_new']
        if (ROI_coordinates['y_new'] > ROI_coordinates['y_temp'])
        else ROI_coordinates['y_temp']
    )
    if ROI_coordinates['clicks'] % 2 == 0:
        # both start and end point get
        sections.append(((x_start, y_start, x_end, y_end), "ROI of Face Detection"))
        return (img, sections)
    else:
        point_width = int(img.shape[0] * 0.05)
        sections.append(
            (
                (
                    ROI_coordinates['x_new'],
                    ROI_coordinates['y_new'],
                    ROI_coordinates['x_new'] + point_width,
                    ROI_coordinates['y_new'] + point_width,
                ),
                "Click second point for ROI",
            )
        )
        return (img, sections)


def process_image(image):
    return image


def interface():
    article = """
         # Zero-shot Template Matching       
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)

        i = gr.Image(source="canvas", shape=(512, 512)).style(width=512, height=512)
        o = gr.Image().style(width=512, height=512)

        with gr.Row():
            input_img = gr.Image(label="Click")
            img_output = gr.AnnotatedImage(
                label="ROI",
                color_map={
                    "ROI of Face Detection": "#9987FF",
                    "Click second point for ROI": "#f44336",
                },
            )
        input_img.select(get_select_coordinates, input_img, img_output)

        if False:
            with gr.Row():
                src = gr.Image(type="pil", source="upload")

            with gr.Row():
                btn_reset = gr.Button("Clear")
                btn_submit = gr.Button("Submit", variant="primary")

            with gr.Row():
                with gr.Column():
                    boxes = gr.components.Image(type="pil", label="boxes")
                with gr.Column():
                    lines = gr.components.Image(type="pil", label="icr")

            with gr.Row():
                with gr.Column():
                    txt = gr.components.Textbox(label="text", max_lines=100)

            with gr.Row():
                with gr.Column():
                    results = gr.components.JSON()

            btn_submit.click(
                process_image, inputs=[src], outputs=[boxes, lines, results, txt]
            )

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
