""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import shutil

import fire
import numpy as np


def split(inputPath, outputPath, datalist):
    nSamples = len(datalist)
    gtFile = os.path.join(outputPath, 'labels.txt')
    with open(gtFile, 'w', encoding='utf-8') as data:
        for i in range(nSamples):
            line = datalist[i].strip('\n')
            imagePath, label = line.split(' ')
            inputImagePath = os.path.join(inputPath, imagePath)
            outputImagePath = os.path.join(outputPath, imagePath)
            if not os.path.exists(inputImagePath):
                print('%s does not exist' % inputImagePath)
                continue
            shutil.copyfile(inputImagePath, outputImagePath)
            data.write(line)
            data.write('\n')

def splitDataset(inputPath, gtFile, outputPath, train_percentage):
    """
    Split dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : output folder where new train/test directory will be outputed
        gtFile     : list of image path and label
        split      : split ration ex: .8-.2
    """

    print('inputPath  = {}'.format(inputPath))
    print('outputPath = {}'.format(outputPath))
    print('gtFile     = {}'.format(gtFile))
    print('split      = {}'.format(train_percentage))

    test_dir = os.path.join(outputPath, 'test')
    train_dir = os.path.join(outputPath, 'train')
    
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    np.random.shuffle(datalist)
    nSamples = len(datalist)
    train_count = int(nSamples * train_percentage)
    test_count = nSamples - train_count

    print('nSamples   = {}'.format(nSamples))
    print('train_rat  = {}'.format(train_count))
    print('test_rat   = {}'.format(test_count))

    split(inputPath, train_dir, datalist[0:train_count])
    split(inputPath, test_dir, datalist[train_count:])
   
if __name__ == '__main__':
    # python ./split_dataset.py --inputPath ~/datasets/icr-finetune --gtFile ~/datasets/icr-finetune/labels.txt --outputPath ~/datasets/icr-finetune-split  --train_percentage .8
    fire.Fire(splitDataset)
