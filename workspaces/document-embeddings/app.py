import os
from typing import List

import gradio as gr
import numpy as np
import torch as torch
from docarray import DocList
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.components.document_taxonomy import DocumentTaxonomySeq2SeqLM
from marie.components.document_taxonomy.base import BaseDocumentTaxonomy
from marie.components.document_taxonomy.datamodel import TaxonomyPrediction
from marie.components.document_taxonomy.qavit_document_taxonomy import (
    QaVitDocumentTaxonomy,
)
from marie.components.document_taxonomy.transformers import (
    DocumentTaxonomyClassification,
)
from marie.components.document_taxonomy.util import (
    group_taxonomies_by_label,
    merge_annotations,
)
from marie.document import TrOcrProcessor
from marie.executor.ner.utils import normalize_bbox
from marie.utils.docs import docs_from_image, frames_from_file

use_cuda = torch.cuda.is_available()
max_input_length = 8192

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=False
# )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_processor(model_type, model_name_or_path) -> BaseDocumentTaxonomy:
    if model_type == 'flan-t5':
        processor = DocumentTaxonomyClassification(
            model_name_or_path=model_name_or_path,
            use_gpu=True,
        )
    elif model_type == 'flan-t5-seq2seq':
        label2id = {"TABLE": 0, "SECTION": 1, "CODES": 2, "OTHER": 3}
        id2label = {id: label for label, id in label2id.items()}
        processor = DocumentTaxonomySeq2SeqLM(
            model_name_or_path=model_name_or_path,
            use_gpu=True,
            k_completions=5,
            id2label=id2label,
        )
    elif model_type == 'qavit':
        processor = QaVitDocumentTaxonomy(
            model_name_or_path=model_name_or_path, use_gpu=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return processor


def build_processorXXXX(model_type, model_name_or_path) -> BaseDocumentTaxonomy:
    if model_type == 'modernberta':
        processor = DocumentTaxonomyClassification(
            model_name_or_path=model_name_or_path,
            use_gpu=True,
        )
    elif model_type == 'flan-t5-seq2seq':
        label2id = {"TABLE": 0, "SECTION": 1, "CODES": 2, "OTHER": 3}
        id2label = {id: label for label, id in label2id.items()}
        processor = DocumentTaxonomySeq2SeqLM(
            model_name_or_path=model_name_or_path,
            use_gpu=True,
            k_completions=5,
            id2label=id2label,
        )
    elif model_type == 'qavit':
        processor = QaVitDocumentTaxonomy(
            model_name_or_path=model_name_or_path, use_gpu=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return processor


def build_ocr_engine():
    text_layout = None
    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    icr_processor = TrOcrProcessor(
        models_dir="/mnt/data/marie-ai/model_zoo/trocr", cuda=use_cuda
    )

    return box_processor, icr_processor, text_layout


#
# document_processor = build_processor('modernberta', 'marie/modernberta-taxonomy-document')
# table_processor = build_processor('modernberta', 'marie/modernberta-taxonomy-document-table')

document_processor = build_processor(
    'flan-t5-seq2seq', 'marie/flan-t5-taxonomy-document-seq2seq'
)
table_processor = build_processor('flan-t5', 'marie/flan-t5-taxonomy-document-table')


box_processor, icr_processor, text_layout = build_ocr_engine()

# Constants for taxonomy keys
DOCUMENT_TAXONOMY_KEY = "taxonomy_document"
SECTION_TAXONOMY_KEY = "taxonomy_section"
DEFAULT_LABEL = "OTHER"


def process_single_taxonomy(
    processor: BaseDocumentTaxonomy,
    documents: DocList,
    metadata: dict,
    taxonomy_key: str,
) -> dict:
    processed_docs = processor.run(documents, [metadata], taxonomy_key=taxonomy_key)
    annotations: List[TaxonomyPrediction] = processed_docs[0].tags[taxonomy_key]
    merged_annotations = merge_annotations(
        annotations, metadata, taxonomy_key, default_label=DEFAULT_LABEL
    )
    grouped_taxonomies = group_taxonomies_by_label(
        merged_annotations["lines"], taxonomy_key
    )
    return {"merged": merged_annotations, "groups": grouped_taxonomies}


def has_matching_lines(lhs_lines: list, rhs_lines: list) -> bool:
    return any(
        s_line["line"] == t_line["line"] for s_line in lhs_lines for t_line in rhs_lines
    )


def process_taxonomies(documents: DocList, metadata: dict):
    print("Processing taxonomies")

    document_result = process_single_taxonomy(
        document_processor, documents, metadata, DOCUMENT_TAXONOMY_KEY
    )
    section_result = process_single_taxonomy(
        table_processor, documents, metadata, SECTION_TAXONOMY_KEY
    )

    # print("Document Results:", document_result["merged"])
    print("Document Taxonomy Groups:", document_result["groups"])
    # print("Section Results:", section_result["merged"])
    print("Section Taxonomy Groups:", section_result["groups"])

    filtered_doc_groups = []
    for doc_group in document_result["groups"]:
        if doc_group["label"] == "TABLE":
            filtered_doc_groups.append(doc_group)

    filtered_section_groups = []
    for doc_group in filtered_doc_groups:
        for section_group in section_result["groups"]:
            if section_group["label"] == "OTHER":
                continue
            if has_matching_lines(doc_group["lines"], section_group["lines"]):
                filtered_section_groups.append(section_group)

    return (
        document_result["groups"],
        filtered_doc_groups,
        filtered_section_groups,
    )  # table_groups#document_result["groups"]  # , section_result["groups"]


def generate_taxonomy_overlay(taxonomy_groups, image):
    image_width = image.size[0]
    aligned_bboxes = [
        [0, bbox[1], image_width, bbox[3]]
        for bbox in (group["bbox"] for group in taxonomy_groups)
    ]
    bbox_labels = [group["label"] for group in taxonomy_groups]
    return visualize_bboxes(
        image, np.array(aligned_bboxes), format="xywh", labels=bbox_labels
    )


def process_image(filename):
    print("Processing image : ", filename)
    image = Image.open(filename).convert("RGB")
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box_processor.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    result, overlay_image = icr_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    # get boxes and words from the result
    words = []
    boxes_norm = []
    print('----------------')
    for lines in result["lines"]:
        print(lines["text"])
    print('----------------')

    for word in result["words"]:
        x, y, w, h = word["box"]
        w_box = [x, y, x + w, y + h]
        words.append(word["text"])
        boxes_norm.append(normalize_bbox(w_box, (image.size[0], image.size[1])))

    print("len words", len(words))
    print("len boxes", len(boxes_norm))

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")

    frames = [image]
    documents = docs_from_image(frames)

    taxonomy_groups, filtered_doc_groups, filtered_section_groups = process_taxonomies(
        documents, result
    )

    taxonomy_overlay_image = generate_taxonomy_overlay(taxonomy_groups, image)
    filtered_doc_overlay_image = generate_taxonomy_overlay(filtered_doc_groups, image)
    filtered_section_overlay_image = generate_taxonomy_overlay(
        filtered_section_groups, image
    )

    return (
        taxonomy_overlay_image,
        filtered_doc_overlay_image,
        filtered_section_overlay_image,
        None,
    )  # to_json(result)


def image_to_gallery(image_src):
    # image_file will be of tempfile._TemporaryFileWrapper type
    filename = image_src.name
    frames = frames_from_file(filename)
    return frames


def print_textbox(x):
    print(x)
    global prefix_text
    prefix_text = x


def interface():
    article = """
         # Document Taxonomy     
        """

    def gallery_click_handler(src_gallery, evt: gr.SelectData):
        selection = src_gallery[evt.index]

        print("selection", selection)
        # filename = selection["name"]
        filename = selection[0]
        return process_image(filename)

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row(variant="compact"):
            with gr.Column():
                src = gr.components.File(
                    type="filepath",  # Corrected type parameter
                    label="Multi-page TIFF/PDF file",
                    file_count="single",
                )
                with gr.Row():
                    btn_reset = gr.Button("Clear")
                    btn_grid = gr.Button("Image-Grid", variant="primary")

            with gr.Column():
                chk_filter_results = gr.Checkbox(
                    label="Filter results",
                    value=False,
                    interactive=True,
                )

                gr.Number(
                    label="Threshold",
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.95,
                    interactive=True,
                    precision=2,
                )

                # chk_apply_overlay.change(
                #     lambda x: update_overlay(x),
                #     inputs=[chk_apply_overlay],
                #     outputs=[],
                # )

        with gr.Row():
            gallery = gr.Gallery(
                label="Image frames",
                show_label=False,
                elem_id="gallery",
                interactive=True,
            )

        btn_grid.click(image_to_gallery, inputs=[src], outputs=gallery)
        btn_reset.click(lambda: src.clear())

        with gr.Row():
            doc_img = gr.components.Image(type="pil", label="Document Taxonomy")
            section_img = gr.components.Image(type="pil", label="Section Taxonomy")
            segment_img = gr.components.Image(type="pil", label="Segment Taxonomy")
        with gr.Row():
            with gr.Column():
                results = gr.components.JSON()

        gallery.select(
            gallery_click_handler,
            inputs=[gallery],
            outputs=[doc_img, section_img, segment_img, results],
        )

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
