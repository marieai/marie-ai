# marie.boxes.BoxProcessorUlimDit

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

#### \_\_init_\_(work_dir: str = '/tmp/boxes', models_dir: str = '/mnt/marie/model_zoo', cuda: bool = False)

Initialize

* **Parameters:**
  * **work_dir** – Working directory
  * **models_dir** – Models directory
  * **cuda** – Is CUDA processing enabled
  * **config** – Configuration map

### Methods

| [`__init__`](#marie.boxes.BoxProcessorUlimDit.__init__)([work_dir, models_dir, cuda])   | Initialize                                                                                                                     |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `describe_handle`()                                                                     | Customized describe handler                                                                                                    |
| `explain_handle`(data_preprocess, raw_data)                                             | Captum explanations handler                                                                                                    |
| `extract_bounding_boxes`(_id, key, img[, psm, ...])                                     | Extract bounding boxes for specific image, try to predict line number representing each bounding box.                          |
| `handle`(data, context)                                                                 | Entry point for default handler. It takes the data from the input request and returns                                          |
| `inference`(data, \*args, \*\*kwargs)                                                   | The Inference Function is used to make a prediction call on the given input request.                                           |
| `initialize`(context)                                                                   | Initialize function loads the model.pt file and initialized the model object.                                                  |
| `postprocess`(data)                                                                     | The post process function makes use of the output from the inference and converts into a Torchserve supported response output. |
| `preprocess`(data)                                                                      | Preprocess function to convert the request input to a tensor(Torchserve supported format).                                     |
| `psm_line`(image)                                                                       | Treat the image as a single text line.                                                                                         |
| `psm_multiline`(image)                                                                  | Treat the image as a single word.                                                                                              |
| `psm_raw_line`(image)                                                                   | Treat the image as a single text line.                                                                                         |
| `psm_sparse`(image[, bbox_optimization, ...])                                           | Find as much text as possible (default).                                                                                       |
| `psm_word`(image)                                                                       | Treat the image as a single word.                                                                                              |
| `unload`()                                                                              | Unload the model from GPU/CPU                                                                                                  |

### Attributes

| `REGISTRY`   |    |
|--------------|----|
