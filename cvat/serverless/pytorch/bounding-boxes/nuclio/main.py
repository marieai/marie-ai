import json
import base64

import yaml
from PIL import Image
import io
from model_handler import ModelHandler

# https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("marie-ai  handler 001")

    data = event.body
    optimize_bboxes = data.get("optimize", False)
    context.logger.info(f"marie-ai optimize_bboxes : {optimize_bboxes}")

    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
    results_json = context.user_data.model.handle(image)

    results_json_test = [
        {
            "confidence": 0.9,
            "name": "bbox",
            "xmin": 0,
            "ymin": 0,
            "xmax": 100,
            "ymax": 100,
        }
    ]

    box_format = "xywh"
    encoded_results = []
    for result in results_json:
        # convert int32 to int
        box = [int(v) for v in result]
        if box_format == "xywh":
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

        encoded_results.append(
            {
                "confidence": str(1.0),
                "label": "bbox",
                "points": box,
                "type": "rectangle",
            }
        )

    context.logger.info("marie-ai  handler 002")
    results = json.dumps(encoded_results)
    context.logger.info(results)

    return context.Response(
        body=results,
        headers={},
        content_type="application/json",
        status_code=200,
    )
