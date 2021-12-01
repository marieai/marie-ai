# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception 
import os
import sys

from icr.memory_dataset import MemoryDataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import typing
import numpy as np
import json
import copy
import cv2
import numpy as np
from PIL import Image
import base64

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from icr.utils import CTCLabelConverter, AttnLabelConverter, Averager
from icr.dataset import hierarchical_dataset, AlignCollate, RawDataset
from icr.model import Model
from icr.single_dataset import SingleDataset

from numpyencoder import NumpyEncoder
from numpycontainer import NumpyContainer
from draw_truetype import drawTrueTypeTextOnImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

from utils.resize_image import resize_image


class Object(object):
    pass

def encodeimg2b64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode('.png', img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text


def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)


def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def icr_debug(opt):
    """
        ICR debug utility
    """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    print('Evaluating on device : %s' % (device))
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # setup data
    AlignCollate_data = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    eval_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_data, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in eval_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                # preds = model(image, text_for_pred, is_train=False)
                preds = model(image, text_for_pred)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./log_eval_result.txt', 'a')
            dashed_line = '-' * 120
            head = f'{"image_path":25s}\t{"predicted_labels":32s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}\n')

            log.close()


def compute_input(image):
    # should be RGB order
    image = image.astype('float32')
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image


class IcrProcessor:
    def __init__(self, work_dir: str = '/tmp/icr', cuda: bool = False) -> None:
        print("ICR processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir

        if True:
            opt = Object()
            opt.Transformation = 'TPS'
            opt.FeatureExtraction = 'ResNet'
            opt.SequenceModeling = 'BiLSTM'
            opt.Prediction = 'Attn'
            opt.saved_model = './models/icr/TPS-ResNet-BiLSTM-Attn-case-sensitive-ft/best_accuracy.pth'
            opt.sensitive = True
            opt.imgH = 32
            opt.imgW = 100
            opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
            opt.rgb = False
            opt.num_fiducial = 20
            opt.input_channel = 1
            opt.output_channel = 512
            opt.hidden_size = 256
            opt.batch_max_length = 48
            opt.batch_size = 2  # FIXME: setting batch size to 1 will cause "TypeError: forward() missing 2 required positional arguments: 'input' and 'text'"
            opt.PAD = True
            opt.workers = 4
            opt.num_gpu = -1
            opt.image_folder = './'

        if False:
            opt = Object()
            opt.Transformation = 'TPS'
            opt.FeatureExtraction = 'ResNet'
            opt.SequenceModeling = 'BiLSTM'
            opt.Prediction = 'Attn'
            opt.saved_model = './models/icr/TPS-ResNet-BiLSTM-Attn/best_accuracy.pth'
            # opt.saved_model = './models/icr/TPS-ResNet-BiLSTM-Attn-case-sensitive/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'
            opt.saved_model = './models/icr/TPS-ResNet-BiLSTM-Attn/TPS-ResNet-BiLSTM-Attn.pth'
            opt.sensitive = False
            opt.imgH = 32
            opt.imgW = 100
            opt.character = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            # opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
            opt.num_fiducial = 20
            opt.input_channel = 1
            opt.output_channel = 512
            opt.hidden_size = 256
            opt.batch_max_length = 25
            opt.batch_size = 2  # Fixme: setting batch size to 1 will cause "TypeError: forward() missing 2 required positional arguments: 'input' and 'text'"
            opt.PAD = True
            opt.rgb = False
            opt.workers = 4
            opt.num_gpu = -1
            opt.image_folder = './'

        self.opt = opt
        self.converter, self.model = self.__load()

        cudnn.benchmark = True
        cudnn.deterministic = True

    def __load(self):
        """ model configuration """
        opt = self.opt

        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        print('Evaluating on device : %s' % (device))
        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)

        # Somehow the model in being still loaded on GPU
        # https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html

        # GPU only
        model = torch.nn.DataParallel(model, device_ids=None).to(device)
        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        if False:
            class WrappedModel(torch.nn.Module):
                def __init__(self, module):
                    super(WrappedModel, self).__init__()
                    self.module = module  # that I actually define.

                def forward(self, x):
                    return self.module(x)

                    # CPU

            model = WrappedModel(model)
            model = model.to(device)
            state_dict = torch.load(opt.saved_model, map_location=device)
            model.load_state_dict(state_dict)

        return converter, model

    def extract_text(self, _id, key, image):
        """Recognize text from a single image.
           Process image via ICR, this is lowlever API, to get more usable results call extract_icr.

        Args:
            _id: Unique Image ID
            key: Unique image key
            image: A pre-cropped image containing characters
        """

        print('ICR processing : {}, {}'.format(_id, key))
        results = self.recognize_from_boxes([image], [0, 0, image.shape[1], image.shape[0]])
        if len(results) == 1:
            r = results[0]
            return r['text'], r['confidence']
        return None, 0

    def recognize_from_boxes(self, image, boxes, **kwargs) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image using lists of bounding boxes.

        Args:
            image: input images, supplied as numpy arrays with shape
                (H, W, 3).
            boxes: A list of boxes to extract
        """
        raise Exception("Not yet implemented")

    def recognize_from_fragments(self, images, **kwargs) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """

        print('ICR processing : recognize_from_boxes via boxes')
        # debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'icr',key,'debug'))
        # output_dir = ensure_exists(os.path.join(self.work_dir,id,'icr',key,'output'))

        opt = self.opt
        model = self.model
        converter = self.converter
        opt.batch_size = 192  #

        # setup data
        AlignCollate_data = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data = MemoryDataset(images=images, opt=opt)

        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_data, pin_memory=True)

        results = []
        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, image_labels in eval_loader:
                print(f'OCR : {image_labels}')
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)

                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)
                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)

                else:
                    preds = model(image, text_for_pred, is_train=False)
                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                log = open(f'./log_eval_result.txt', 'a')
                dashed_line = '-' * 120
                head = f'{"key":25s}\t{"predicted_labels":32s}\tconfidence score'

                print(f'{dashed_line}\n{head}\n{dashed_line}')
                log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                for img_name, pred, pred_max_prob in zip(image_labels, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    # get value from the TensorFloat
                    confidence = confidence_score.item()
                    txt = pred
                    results.append({
                        "confidence": confidence,
                        "text": txt,
                        "id": img_name
                    })

                    print(f'{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}')
                    log.write(f'{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}\n')
                log.close()

        return results

    def recognize(self, _id, key, img, boxes, image_fragments, lines):
        """Recognize text from multiple images.
        Args:
            id: Unique Image ID
            key: Unique image key/region for the extraction
            img: A pre-cropped image containing characters
        """
        print(f'ICR recognize : {_id}, {key}')
        assert len(boxes) == len(image_fragments), "You must provide the same number of box groups as images."

        shape = img.shape
        overlay_image = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
        debug_dir = ensure_exists(os.path.join('/tmp/icr', _id))
        debug_all_dir = ensure_exists(os.path.join('/tmp/icr', 'fields', key))

        meta = {
            'imageSize': {'width': img.shape[1], 'height': img.shape[0]},
            'lang': 'en'
        }

        words = []
        max_line_number = 0
        results = self.recognize_from_fragments(image_fragments)

        for i in range(len(boxes)):
            box, fragment, line = boxes[i], image_fragments[i], lines[i]
            # txt, confidence = self.extract_text(id, str(i), fragment)
            extraction = results[i]
            txt = extraction['text']
            confidence = extraction['confidence']
            print('Processing [box, line, txt, conf] : {}, {}, {}, {}'.format(box, line, txt, confidence))
            conf_label = f'{confidence:0.4f}'
            txt_label = txt

            payload = dict()
            payload['id'] = i
            payload['text'] = txt
            payload['confidence'] = round(confidence, 4)
            payload['box'] = box
            payload['line'] = line
            payload['fragment_b64'] = encodeimg2b64(fragment)

            words.append(payload)

            if line > max_line_number:
                max_line_number = line

            overlay_image = drawTrueTypeTextOnImage(overlay_image, txt_label, (box[0], box[1] + box[3] // 2), 18,
                                                    (139, 0, 0))
            overlay_image = drawTrueTypeTextOnImage(overlay_image, conf_label, (box[0], box[1] + box[3]), 10,
                                                    (0, 0, 255))

        savepath = os.path.join(debug_dir, f'{key}-icr-result.png')
        imwrite(savepath, overlay_image)

        savepath = os.path.join(debug_all_dir, f'{_id}.png')
        imwrite(savepath, overlay_image)

        line_ids = np.empty((max_line_number), dtype=object)
        words = np.array(words)

        for i in range(0, max_line_number):
            current_lid = i + 1
            word_ids = []
            box_picks = []
            word_picks = []

            for word in words:
                lid = word['line']
                if lid == current_lid:
                    word_ids.append(word['id'])
                    box_picks.append(word['box'])
                    word_picks.append(word)

            box_picks = np.array(box_picks)
            word_picks = np.array(word_picks)

            x1 = box_picks[:, 0]
            idxs = np.argsort(x1)
            aligned_words = word_picks[idxs]
            _w = []
            _conf = []

            for wd in aligned_words:
                _w.append(wd['text'])
                _conf.append(wd['confidence'])

            text = ' '.join(_w)

            min_x = box_picks[:, 0].min()
            min_y = box_picks[:, 1].min()
            max_w = box_picks[:, 2].max()
            max_h = box_picks[:, 3].max()
            bbox = [min_x, min_y, max_w, max_h]

            line_ids[i] = {
                'line': i + 1,
                'wordids': word_ids,
                'text': text,
                'bbox': bbox,
                'confidence': round(np.average(_conf), 4)
            }

        result = {
            'meta': meta,
            'words': words,
            'lines': line_ids,
        }

        print(result)

        with open('/tmp/icr/data.json', 'w') as f:
            json.dump(result, f,  sort_keys=True,  separators=(',', ': '), ensure_ascii=False, indent=4, cls=NumpyEncoder)

        print('------ Extraction ------------')
        for line in line_ids:
            txt = line['text']
            print(f' >> {txt}')

        return result, overlay_image
