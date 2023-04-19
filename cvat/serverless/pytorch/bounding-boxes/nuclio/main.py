import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("marie-ai  handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
    results_json = context.user_data.model.handle(image)
    results_json = [
        {
            'confidence': 0.9,
            'name': 'bbox',
            'xmin': 0,
            'ymin': 0,
            'xmax': 100,
            'ymax': 100,
        }
    ]

    encoded_results = []
    for result in results_json:
        encoded_results.append(
            {
                'confidence': result['confidence'],
                'label': result['name'],
                'points': [
                    result['xmin'],
                    result['ymin'],
                    result['xmax'],
                    result['ymax'],
                ],
                'type': 'rectangle',
            }
        )

    return context.Response(
        body=json.dumps(encoded_results),
        headers={},
        content_type='application/json',
        status_code=200,
    )
