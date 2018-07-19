"""
Test the trained model on images
"""
from __future__ import print_function, division, absolute_import

import argparse
import os
import json
import cv2
import numpy as np
import torch as th
from torch.autograd import Variable
from tqdm import tqdm
from Pre.constants import INPUT_WIDTH, INPUT_HEIGHT, RIGHT_KEY, LEFT_KEY, EXIT_KEYS, ROI,MAX_WIDTH, UP_KEY, DOWN_KEY, TIME_GAP, TOTAL_FRAME
from Pre.utils import preprocessImage, transformPrediction, loadLabels
from Pre.models import ConvolutionalNetwork, CNN_RNN, CNN_LSTM

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
parser.add_argument('-w', '--weights', help='Saved weights', default="cnn_model_25_tmp.pth", type=str)
parser.add_argument('-m', '--model', help='Model type', default="cnn", type=str)
parser.add_argument('-t', '--time_gap', help='Time gap', type=int, default=25)
args = parser.parse_args()

train_labels, val_labels, test_labels, labels = loadLabels(args.folder, args.model)
ONE_IMG_ONLY = False
# images = list(labels.keys())
# TODO: add support for folders without labels
if True:
    images = [f for f in os.listdir(args.folder) if f.endswith('.png')]

images.sort(key=lambda name: int(name.split('.png')[0]))

idx_val = set(val_labels.keys())
idx_test = set(test_labels.keys())
numChannel = 3
if ONE_IMG_ONLY:
    numChannel = 3
else:
    numChannel = 6
current_idx = 0
if args.model == "CNN_RNN":
    model = CNN_RNN(num_channel=numChannel)
elif args.model == "CNN_LSTM":
    model = CNN_LSTM(num_channel=numChannel)
else: model = ConvolutionalNetwork(num_channel=numChannel, num_output=2)
model.load_state_dict(th.load(args.weights))
model.eval()# IMPORTANT, if drop rate is not 0, without this prediction is bad
# TODO: compute val and test error
res = {}

print("start testing ...")
for current_idx in tqdm(range(len(images))):
    # if take only one image for prediction
    if ONE_IMG_ONLY:
        name = images[current_idx]
        image = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))
        imgs = preprocessImage(image, INPUT_WIDTH, INPUT_HEIGHT)
        imgs = imgs.transpose((2, 0, 1)).astype(np.float32)
    # take two images as input
    else:
        imgs = []
        for i in range(2):
            try:
                image = images[current_idx+i]
                image = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))
                im = preprocessImage(image, INPUT_WIDTH, INPUT_HEIGHT)
                imgs.append(im)
            except IndexError:
                imgs.append(imgs[0])
                continue
        imgs = np.dstack(imgs)
        imgs = imgs.transpose((2, 0, 1)).astype(np.float32)
    predictions = model(Variable(th.from_numpy(imgs[None])))[0].data.numpy()
    x, y = transformPrediction(predictions)
    ls = []
    ls.append(float(x))
    ls.append(float(y))
    res.update({str(int(images[current_idx].replace('.png',""))+args.time_gap):ls})

# print(res)
print("test finished !")
jsObj = json.dumps(res)

fileObject = open('./Pre/3dmodel/predictions_'+args.model+"_"+str(args.time_gap)+'.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print('Done!')
