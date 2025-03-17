### Stolen from https://huggingface.co/spaces/pierreguillou/tatr-demo/blob/main/app.py

import io

import gradio as gr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForObjectDetection

device = "cuda" if torch.cuda.is_available() else "cpu"


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# load table detection model
# processor = TableTransformerImageProcessor(max_size=800)
model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)

# load table structure recognition model
# structure_processor = TableTransformerImageProcessor(max_size=1000)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append(
                {
                    'label': class_label,
                    'score': float(score),
                    'bbox': [float(elem) for elem in bbox],
                }
            )

    return objects


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    return image


def visualize_detected_tables(img, det_tables):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor='none',
            facecolor=facecolor,
            alpha=0.1,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle='-',
            alpha=alpha,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle='-',
            hatch=hatch,
            alpha=0.2,
        )
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [
        Patch(
            facecolor=(1, 0, 0.45),
            edgecolor=(1, 0, 0.45),
            label='Table',
            hatch='//////',
            alpha=0.3,
        ),
        Patch(
            facecolor=(0.95, 0.6, 0.1),
            edgecolor=(0.95, 0.6, 0.1),
            label='Table (rotated)',
            hatch='//////',
            alpha=0.3,
        ),
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.02),
        loc='upper center',
        borderaxespad=0,
        fontsize=10,
        ncol=2,
    )
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    return fig


def detect_and_crop_table(image):
    # prepare image for the model
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # postprocess to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    # visualize
    # fig = visualize_detected_tables(image, detected_tables)
    # image = fig2img(fig)

    # crop first detected table out of image
    cropped_table = image.crop(detected_tables[0]["bbox"])

    return cropped_table


def recognize_table(image):
    # prepare image for the model
    # pixel_values = structure_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = structure_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # postprocess to get individual elements
    id2label = structure_model.config.id2label
    id2label[len(structure_model.config.id2label)] = "no object"
    cells = outputs_to_objects(outputs, image.size, id2label)

    header_cells = []
    for cell in cells:
        if cell["label"] not in ["table header", "table column"]:
            continue
        header_cells.append(cell)

    print("----------")
    for i, cell in enumerate(header_cells):
        print(f"{i} > {cell}")

    # visualize cells on cropped table
    draw = ImageDraw.Draw(image)
    for cell in cells:
        # filter to only show table headers
        print("cell", cell)
        # if cell["label"] not in ["table header", "table column header"]:
        #     continue
        fill_color = "red" if cell["label"] == "table column header" else "blue"
        draw.rectangle(cell["bbox"], outline=fill_color)
        draw.text((cell["bbox"][0], cell["bbox"][1]), cell["label"], fill=fill_color)

    return image, cells


def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column['bbox'][0],
            row['bbox'][1],
            column['bbox'][2],
            row['bbox'][3],
        ]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append(
            {'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)}
        )

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_table):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    ui_idx = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            # crop cell out of image
            print('cell-info', cell)
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            # # save the cell image
            # cell_image_pil = Image.fromarray(cell_image)
            # cell_image_pil.save(f"./debug/cell_{ui_idx}.png")
            # apply OCR
            result = "Hello"  # reader.readtext(np.array(cell_image))
            row_text.append(result)

            ui_idx += 1

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[str(idx)] = row_text

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for idx, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[str(idx)] = row_data

    # # write to csv
    # with open('output.csv','w') as result_file:
    #     wr = csv.writer(result_file, dialect='excel')

    #     for row, row_text in data.items():
    #         wr.writerow(row_text)

    # # return as Pandas dataframe
    # df = pd.read_csv('output.csv')
    return data


def dump_cells(cropped_table, table_data):
    header = [entry for entry in table_data if entry['label'] == 'table column header']

    for ui_idx, cell in enumerate(header):
        print("cropped cell", cell)
        # crop cell out of image
        cell_image = np.array(cropped_table.crop(cell["bbox"]))
        # save the cell image
        cell_image_pil = Image.fromarray(cell_image)
        cell_image_pil.save(f"./debug/header_{ui_idx}.png")


def process_document(image):
    cropped_table = detect_and_crop_table(image)
    # dump cropped_table to disk
    cropped_table.save("./debug/cropped_table.png")

    image, cells = recognize_table(cropped_table)
    cell_coordinates = get_cell_coordinates_by_row(cells)

    dump_cells(cropped_table, cells)
    data = apply_ocr(cell_coordinates, image)

    return image, data


title = "Demo: table detection & recognition with Table Transformer (TATR)."
description = """Demo for table extraction with the Table Transformer. First, table detection is performed on the input image using https://huggingface.co/microsoft/table-transformer-detection,
after which the detected table is extracted and https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all is leveraged to recognize the individual rows, columns and cells. OCR is then performed per cell, row by row."""

app = gr.Interface(
    fn=process_document,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil", label="Detected table"), gr.JSON(label="JSON")],
)
app.queue()
app.launch(debug=True)
