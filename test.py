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

from constants import INPUT_WIDTH, INPUT_HEIGHT, RIGHT_KEY, LEFT_KEY, EXIT_KEYS, ROI,MAX_WIDTH, UP_KEY, DOWN_KEY
from .utils import preprocessImage, transformPrediction, loadLabels
from .models import ConvolutionalNetwork, RNN

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
parser.add_argument('-w', '--weights', help='Saved weights', default="cnn_model_tmp.pth", type=str)
parser.add_argument('-m', '--model', help='Model type', default="cnn", type=str)
args = parser.parse_args()

current_idx = 0
if args.model == "RNN":
    model = RNN(4, 3*INPUT_HEIGHT*INPUT_WIDTH)
else: model = ConvolutionalNetwork(num_output=3)
model.load_state_dict(th.load(args.weights))
# modified by zhi IMPORTANT, if drop rate is not 0, without this prediction is bad
model.eval()

train_labels, val_labels, test_labels, labels = loadLabels(args.folder)

# images = list(labels.keys())
# TODO: add support for folders without labels
if True:
    images = [f for f in os.listdir(args.folder) if f.endswith('.png')]
    # labels = {}

images.sort(key=lambda name: int(name.split('.png')[0]))

idx_val = set(val_labels.keys())
idx_test = set(test_labels.keys())

# TODO: compute val and test error
res = {}
h_state = None
for current_idx in range(len(images)):
    name = images[current_idx]
    image = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))
    # image = cv2.flip(image, 1)
    # Image from train/validation/test set ?
    # text = "train " + images[current_idx]
    # if name in idx_val:
    #     text = "val " + images[current_idx]
    # elif name in idx_test:
    #     text = "test " + images[current_idx]
    # cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    # print(current_idx)

    im = preprocessImage(image, INPUT_WIDTH, INPUT_HEIGHT)

    im = im.transpose((2, 0, 1)).astype(np.float32)
    #predictions = model(Variable(th.from_numpy(im[None]), volatile=True))[0].data.numpy()
    if args.model == "RNN":
        # global h_state
        predictions, h_state = model(Variable(th.from_numpy(im[None])), h_state)[0].data.numpy()
    else: predictions = model(Variable(th.from_numpy(im[None])))[0].data.numpy()
    x, y, z = transformPrediction(predictions)
    # print(type(x))
    # rot = list(degrees(a) for a in rot)
    ls = []
    ls.append(float(x))
    ls.append(float(y))
    ls.append(float(z))
    # for i in [x, y, z]:
    #     if abs(i) > 1000:
    #         i = 1000*i/abs(i)
    # res.update({str(current_idx):map(float,[x, y, z])})
    res.update({images[current_idx].replace('.png',""):ls})
    # if labels.get(images[current_idx]) is not None:
    #     true_labels = np.array(labels[images[current_idx]])
    # else:
    #     true_labels = None

    # Draw prediction and label
    # cv2.line(image, (int(x-MAX_WIDTH*sint/2), int(y+ MAX_WIDTH*cost/2)), (int(x+MAX_WIDTH*sint/2), int(y-MAX_WIDTH*cost/2)), color=(176, 114, 76),thickness=3)
    # if true_labels is not None:
    #     cv2.line(image, (int(true_labels[0]-MAX_WIDTH*true_labels[3]/2), int(true_labels[1]+ MAX_WIDTH*true_labels[2]/2)), (int(true_labels[0]+MAX_WIDTH*true_labels[3]/2), int(true_labels[1]- MAX_WIDTH*true_labels[2]/2)),color=(104, 168, 85),thickness=3)
    # cv2.imshow('Prediction', image)

    # modified by zhi
    # cv2.line(image, (int(x-MAX_WIDTH*sint/2), int(y+ MAX_WIDTH*cost/2)), (int(x+MAX_WIDTH*sint/2), int(y-MAX_WIDTH*cost/2)), color=(176, 114, 76),thickness=3)
    # if true_labels is not None:
        # cv2.line(image, (int(true_labels[0]-MAX_WIDTH*true_labels[3]/2), int(true_labels[1]+ MAX_WIDTH*true_labels[2]/2)), (int(true_labels[0]+MAX_WIDTH*true_labels[3]/2), int(true_labels[1]- MAX_WIDTH*true_labels[2]/2)),color=(104, 168, 85),thickness=3)
        # modified by zhi
        # cv2.line(image, (int(true_labels[0]-MAX_WIDTH*true_labels[3]/2), int(true_labels[1]-60+ MAX_WIDTH*true_labels[2]/2)), (int(true_labels[0]+MAX_WIDTH*true_labels[3]/2), int(true_labels[1]-60- MAX_WIDTH*true_labels[2]/2)),color=(104, 168, 85),thickness=3)
    # cv2.imshow('Prediction', image)

    # r = ROI
    # im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # cv2.imshow('Crop', im_cropped)

    # key = cv2.waitKey(0) & 0xff
    # #print(key)
    # if key in EXIT_KEYS:
    #     cv2.destroyAllWindows()
    #     break
    # elif key in [LEFT_KEY, RIGHT_KEY, UP_KEY, DOWN_KEY]:
    #     if key == RIGHT_KEY:
    #         current_idx += 1
    #     elif key == LEFT_KEY:
    #         current_idx -= 1
    #     elif key == UP_KEY:
    #         current_idx += 10
    #     elif key == DOWN_KEY:
    #         current_idx -= 10
    #     current_idx = np.clip(current_idx, 0, len(images) - 1)
print(res)
# print(type(res["1"]))
# jsObj = json.dumps(res, default=lambda x: None)
jsObj = json.dumps(res)

fileObject = open('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/test.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print('Done!')
