import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler

# https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("marie-ai  handler 001")

    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
    results_json = context.user_data.model.handle(image)

    context.logger.info("marie-ai  handler 002")
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

    results = json.dumps(encoded_results)
    context.logger.info(results)

    return context.Response(
        body=results,
        headers={},
        content_type="application/json",
        status_code=200,
    )
