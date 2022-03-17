""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import glob
import os

import fire
import numpy as np

import cv2
import lmdb


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def imwrite(path, img):
    try:
        print(path)
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        # print( datalist[i].strip('\n').split(' ', maxsplit=2))
        # imagePath, label = datalist[i].strip('\n').split(' ', maxsplit=2)
        # # imagePath, label = datalist[i].strip('\n').split('\t')
        # imagePath = os.path.join(inputPath, imagePath)
        line = datalist[i]
        start = line.find(' ')
        imagePath = line[: start] 
        label = line[start:].lstrip()
        label = label.replace('\n', '')

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue
        # imagePath='/home/greg/dev/datasets/generated/train/symbol-print-w3-c5/'+imagePath
        # imagePath='/home/greg/dev/datasets/generated/test/word-print-w1-c0/'+imagePath
        imagePath = os.path.join(inputPath, imagePath)

        # from resize_image import resize_image
        # MAX_IMAGE_SIZE_LINE = (60, 250)
        # MAX_IMAGE_SIZE_LINE = (48, 140)
        # MAX_IMAGE_SIZE_LINE = (60, 240)
        # im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # im, _ = resize_image(im, MAX_IMAGE_SIZE_LINE)
        # gen_path= '/tmp/debug/snip_%s_src.png' % (i)
        # imwrite(gen_path , im)
        # imagePath = gen_path

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def create_labels_file(inputPath, gtFile, outputPath):
    "Created ground truth labels file in the root directory of the input"
    root = inputPath
    _labels = ''
    # labels_src = os.path.join(root, labels)
    label_files = glob.glob(root + '/**/labels.txt', recursive = True)
    print(label_files)

    class BreakOutOfALoop(Exception): pass  
    _parent = inputPath.split("/")[-1]

    # os.path.join(inputPath, 'labels-aggro.txt')
    with open(gtFile, 'w', encoding='utf-8') as f_aggro:
        for labels_src in label_files:
            try:
                parent = os.path.dirname(labels_src)
                pid = parent.split("/")[-1]
                idx = 0
                with open(labels_src) as fp:
                    for line in fp.readlines():
                        if line == '\n|\r':
                            continue
                        start = line.find(' ')
                        fname = line[:start] 
                        value = line[start:].lstrip()
                        # value = value.replace('\n', '')
                        # print('Name = {} : {}'.format(fname, value))
                        if _parent == pid:
                            img_in = "{}".format(fname)
                        else:
                            img_in = os.path.join(pid, "{}".format(fname))                              
                        idx=idx+1
                        f_aggro.write(img_in)
                        f_aggro.write(' ')
                        f_aggro.write(value)
            except BreakOutOfALoop:
                    break

if __name__ == '__main__':

    fire.Fire(create_labels_file)
    fire.Fire(createDataset)
