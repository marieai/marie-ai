from flask_restful import Resource, reqparse, request
from logger import create_info_logger
from utils.utils import current_milli_time, ensure_exists
import werkzeug
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import processors
import json

import hashlib
from numpyencoder import NumpyEncoder
import base64
from skimage import io

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff', 'tif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encodeToBase64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode('.png', img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img

logger = create_info_logger("marie", "marie.log")

class IcrExtractAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        super(IcrExtractAPI, self).__init__()
        self.box_processor = processors.box_processor
        self.icr_processor = processors.icr_processor

        self.show_error = True # show predition errors
        logger.info('IcrExtractAPI inited')

    def post(self, queue_id: str):
        """ICR payload to process
           Process image via ICR, this is low level API, to get more usable results call extract_icr.

        Args:
            queue_id: Unique queue to tie the extraction to
        """

        print('ICR processing')

        parser = reqparse.RequestParser()
        parser.add_argument('file',
                            type=werkzeug.datastructures.FileStorage,
                            location='files',
                            required=True,
                            help='provide a file')

        logger.info(f'Starting ICR processing request', extra={"session": queue_id})

        args = parser.parse_args()
        # read like a stream
        file = args['file']

        if file and not allowed_file(file.filename):
            return {'error': 'Invalid file'}, 400

        stream = file.read()

        m = hashlib.sha256()
        m.update(stream)
        checksum = m.hexdigest()
        upload_dir = ensure_exists(f'/tmp/marie/{queue_id}')
        ext = file.filename.rsplit('.', 1)[1].lower()
        
        try:
            # convert to numpy array
            npimg = np.fromstring(stream, np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            tmp_file = f"{upload_dir}/{checksum}.{ext}"
            cv2.imwrite(tmp_file, img)

            img = loadImage(tmp_file)
            # snippet = img
            h = img.shape[0]
            w = img.shape[1]
            # allow for small padding around the component
            padding = 4
            overlay = np.ones((h+padding*2, w+padding*2, 3), dtype=np.uint8)*255
            overlay[padding:h+padding, padding:w+padding] = img

            cv2.imwrite('/tmp/marie/overlay.png', overlay)

            boxes, img_fragments, lines, _ = self.box_processor.extract_bounding_boxes(
                queue_id, checksum, overlay)
            result, overlay_image = self.icr_processor.recognize(
                queue_id, checksum, overlay, boxes, img_fragments, lines)

            cv2.imwrite('/tmp/marie/overlay_image.png', overlay_image)

            result['overlay_b64'] = encodeToBase64(overlay_image)
            serialized = json.dumps(result, sort_keys=True,  separators=(',', ': '), ensure_ascii=False, indent=2, cls=NumpyEncoder)

            return serialized, 200
        except BaseException as error:
            print(str(error))
            if self.show_error:
                return {"error": str(error)}, 500
            else:
                return {"error": 'inference exception'}, 500                