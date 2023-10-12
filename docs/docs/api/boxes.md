# marie.boxes

| [`BoxProcessorUlimDit`](generated/marie.boxes.BoxProcessorUlimDit.md#marie.boxes.BoxProcessorUlimDit)   | Document text box processor using DIT model from ULIM.   |
|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`PSMode`](generated/marie.boxes.PSMode.md#marie.boxes.PSMode)                                          | Page Segmentation Modes                                  |

### *class* marie.boxes.PSMode(value)

Page Segmentation Modes

* WORD = Treat the image as a single word.
* SPARSE = Sparse text. Find as much text as possible in no particular order.
* LINE = Treat the image as a single text line.
* RAW_LINE =  Raw line. Treat the image as a single text line, NO bounding box detection performed.
* MULTI_LINE = Multiline. Treat the image as multiple text lines, NO bounding box detection performed.

### *class* marie.boxes.BoxProcessorUlimDit(work_dir: str = '/tmp/boxes', models_dir: str = '/mnt/marie/model_zoo', cuda: bool = False)

Document text box processor using DIT model from ULIM.

EXAMPLE USAGE

```python
from marie.boxes import BoxProcessorUlimDit
from marie.boxes.box_processor import PSMode

box = BoxProcessorUlimDit(
    models_dir="../../model_zoo/unilm/dit/text_detection",
    cuda=True,
)
(
    boxes,
    fragments,
    lines,
    _,
    lines_bboxes,
) = box.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

bboxes_img = visualize_bboxes(image, boxes, format="xywh")
lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")
```

<!-- .. autoclass:: BoxProcessorUlimDit -->
<!-- :show-inheritance: -->
<!-- :no-index: -->
<!-- :members: -->
<!-- :undoc-members: -->
<!-- .. autoclass:: BoxProcessorUlimDit -->
<!-- :members: -->
<!-- :undoc-members: -->
<!-- :show-inheritance: -->
