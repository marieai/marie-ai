# marie.document.TesseractOcrProcessor

### *class* marie.document.TesseractOcrProcessor(work_dir: str = '/tmp/icr', models_dir: str = '/mnt/marie/model_zoo/tessdata', cuda: bool = True)

A processor which uses tesseract OCR to process

#### \_\_init_\_(work_dir: str = '/tmp/icr', models_dir: str = '/mnt/marie/model_zoo/tessdata', cuda: bool = True)

### Methods

| [`__init__`](#marie.document.TesseractOcrProcessor.__init__)([work_dir, models_dir, cuda])   |                                                                                                                                |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `describe_handle`()                                                                          | Customized describe handler                                                                                                    |
| `explain_handle`(data_preprocess, raw_data)                                                  | Captum explanations handler                                                                                                    |
| `extract_text`(_id, key, image)                                                              | Recognize text from a single image.                                                                                            |
| `handle`(data, context)                                                                      | Entry point for default handler. It takes the data from the input request and returns                                          |
| `inference`(data, \*args, \*\*kwargs)                                                        | The Inference Function is used to make a prediction call on the given input request.                                           |
| `initialize`(context)                                                                        | Initialize function loads the model.pt file and initialized the model object.                                                  |
| `invoke_tesseract`(index, image)                                                             |                                                                                                                                |
| `is_available`()                                                                             | Returns True if the processor is available for use                                                                             |
| `postprocess`(data)                                                                          | The post process function makes use of the output from the inference and converts into a Torchserve supported response output. |
| `preprocess`(data)                                                                           | Preprocess function to convert the request input to a tensor(Torchserve supported format).                                     |
| `recognize`(_id, key, img, boxes, fragments, lines)                                          | Recognize text from multiple images.                                                                                           |
| `recognize_from_boxes`(image, boxes, \*\*kwargs)                                             | Recognize text from image using lists of bounding boxes.                                                                       |
| `recognize_from_fragments`(images, \*\*kwargs)                                               | Recognize text from image fragments                                                                                            |
| `unload`()                                                                                   | Unload the model from GPU/CPU                                                                                                  |

### Attributes

| `REGISTRY`   |    |
|--------------|----|
