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

from constants import INPUT_WIDTH, INPUT_HEIGHT, RIGHT_KEY, LEFT_KEY, EXIT_KEYS, ROI,MAX_WIDTH, UP_KEY, DOWN_KEY, TIME_GAP, TOTAL_FRAME
from Pre.utils import preprocessImage, transformPrediction, loadLabels
from Pre.models import ConvolutionalNetwork, CNN_RNN, CNN_LSTM

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
parser.add_argument('-w', '--weights', help='Saved weights', default="cnn_model_tmp.pth", type=str)
parser.add_argument('-m', '--model', help='Model type', default="cnn", type=str)
parser.add_argument('-t', '--time_gap', help='Time gap', type=int)
args = parser.parse_args()

current_idx = 0
if args.model == "CNN_RNN":
    model = CNN_RNN()
elif args.model == "CNN_LSTM":
    model = CNN_LSTM()
else: model = ConvolutionalNetwork(num_output=2)
model.load_state_dict(th.load(args.weights))
# modified by zhi IMPORTANT, if drop rate is not 0, without this prediction is bad
model.eval()

train_labels, val_labels, test_labels, labels = loadLabels(args.folder, args.model)

# images = list(labels.keys())
# TODO: add support for folders without labels
if True:
    images = [f for f in os.listdir(args.folder) if f.endswith('.png')]

images.sort(key=lambda name: int(name.split('.png')[0]))

idx_val = set(val_labels.keys())
idx_test = set(test_labels.keys())

# TODO: compute val and test error
res = {}
h_state = None
print("start testing ...")
for current_idx in range(len(images)):
    name = images[current_idx]
    image = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))
    im = preprocessImage(image, INPUT_WIDTH, INPUT_HEIGHT)
    im = im.transpose((2, 0, 1)).astype(np.float32)
    if False: #args.model == "CNN_RNN":
        # global h_state
        predictions, h_state = model(Variable(th.from_numpy(im[None])), h_state)[0].data.numpy()
    else: predictions = model(Variable(th.from_numpy(im[None])))[0].data.numpy()
    x, y = transformPrediction(predictions)
    ls = []
    ls.append(float(x))
    ls.append(float(y))
    # res.update({str(current_idx):map(float,[x, y, z])})
    # res.update({images[current_idx].replace('.png',"")+str(TIME_GAP):ls})
    res.update({str(int(images[current_idx].replace('.png',""))+TIME_GAP):ls})
    # print(int(images[current_idx].replace('.png',""))+TIME_GAP, ls)

# print(res)
print("test finished !")
# print(type(res["1"]))
# jsObj = json.dumps(res, default=lambda x: None)
jsObj = json.dumps(res)

fileObject = open('./Pre/3dmodel/predictions_'+args.model+"_"+str(args.time_gap)+'.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print('Done!')
