from __future__ import print_function, division, absolute_import

import json
import re
import cv2
import numpy as np
import torch as th
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Pre.constants import MAX_WIDTH, MAX_HEIGHT, ROI, INPUT_HEIGHT, INPUT_WIDTH, SPLIT_SEED
from Pre.data_aug import imgTransform
from PIL import Image
import scipy.misc
import matplotlib.image as mpimg

def adjustLearningRate(optimizer, epoch, n_epochs, lr_init, batch,
                         n_batch, method='cosine'):
    """
    :param optimizer: (PyTorch Optimizer object)
    :param epoch: (int)
    :param n_epochs: (int)
    :param lr_init: (float)
    :param batch: (int)
    :param n_batch: (int)
    :param method: (str)
    """
    if method == 'cosine':
        T_total = n_epochs * n_batch
        T_cur = (epoch % n_epochs) * n_batch + batch
        lr = 0.5 * lr_init * (1 + np.cos(np.pi * T_cur / T_total))
    elif method == 'multistep':
        lr, decay_rate = lr_init, 0.7
        if epoch >= n_epochs * 0.75:
            lr *= decay_rate ** 2
        elif epoch >= n_epochs * 0.5:
            lr *= decay_rate
    # else:
    #     # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    #     lr = lr_init * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def preprocessImage(image, width, height):
    """
    Preprocessing script to convert image into neural net input array
    :param image: (cv2 image)
    :param width: (int)
    :param height: (int)
    :return: (numpy tensor)
    """
    # Crop the image
    r = ROI
    image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # The resizing is a bottleneck in the computation
    x = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Normalize
    x = x / 255.
    x -= 0.5
    x *= 2
    return x

def transformPrediction(y):
    margin_left, margin_top, _, _ = ROI
    points = y.flatten()
    x = points[0]
    y = points[1]
    return x, y

def loadLabels(folder, modelType):
    if not folder.endswith('/'):
        folder += '/'
    labels = json.load(open(folder + 'labels.json'))
    # labels2 = labels.copy()
    # lenght = len(labels2)
    # change keys' number in the new labels2
    # labels2 = dict((str(int(key.replace('"',''))+lenght),value) for (key, value) in labels2.items())
    # update labels(combine labels and labels2)
    # labels.update(labels2)
    # json.dump(labels,open('labels2.json','w'))
    images = list(labels.keys())
    images.sort(key=lambda name: int(name.strip().replace('"',"")))
    if False:# modelType == "CNN_LSTM":
        train_keys = images[:int(round(0.6*len(images)))]
        val_keys = images[int(round(0.6*len(images))):int(round(0.8*len(images)))]
        test_keys = images[int(round(0.8*len(images))):len(images)]
        # train_keys = images[:int(round(0.3*len(images)))]
        # train_keys += images[int(round(0.5*len(images))):int(round(0.8*len(images)))]
        # val_keys = images[int(round(0.3*len(images))):int(round(0.4*len(images)))]
        # val_keys += images[int(round(0.8*len(images))):int(round(0.9*len(images)))]
        # test_keys = images[int(round(0.9*len(images))):len(images)]
        # test_keys += images[int(round(0.4*len(images))):int(round(0.5*len(images)))]
    else:
        train_keys, tmp_keys = train_test_split(list(labels.keys()), test_size=0.01, random_state=SPLIT_SEED)
        val_keys, test_keys = train_test_split(tmp_keys, test_size=0.5, random_state=SPLIT_SEED)

    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}
    test_labels = {key: labels[key] for key in test_keys}

    print("{} images".format(len(labels)))
    return train_labels, val_labels, test_labels, labels
"""
take one image only as input
"""
class JsonDatasetOrigin(Dataset):
    def __init__(self, labels, folder="", preprocess=False, random_flip=0.0, swap=False):
        self.keys = list(labels.keys())
        self.labels = labels.copy()
        self.folder = folder
        self.preprocess = preprocess
        self.random_flip = random_flip
        self.swap = swap
        self.numChannel = 3

    def __getitem__(self, index):
        """
        :param index: (int)
        :return: (PyTorch Tensor, PyTorch Tensor)
        """
        image = self.keys[index]
        image += ".png"
        margin_left, margin_top = 0, 0
        im = cv2.imread(self.folder + image)

        # Crop the image and normalize it
        if self.preprocess:
            margin_left, margin_top, _, _ = ROI
            im = preprocessImage(im, INPUT_WIDTH, INPUT_HEIGHT)
        image = image.replace(".png","")
        labels = np.array(self.labels[image]).astype(np.float32)
        y = labels.flatten().astype(np.float32)
        self.numChannel = im.shape[2]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1)).astype(np.float32)
        return th.from_numpy(im), th.from_numpy(y)

    def __len__(self):
        return len(self.keys)
"""
take two images as inputs
"""
class JsonDataset(Dataset):
    def __init__(self, labels, folder="", preprocess=False, random_flip=0.0, swap=False):
        self.keys = list(labels.keys())
        self.labels = labels.copy()
        self.folder = folder
        self.preprocess = preprocess
        self.random_flip = random_flip
        self.swap = swap
        self.numChannel = 6

    def __getitem__(self, index):
        """
        :param index: (int)
        :return: (PyTorch Tensor, PyTorch Tensor)
        """
        images = []
        image = self.keys[index]
        labels = np.array(self.labels[image]).astype(np.float32)
        y = labels.flatten().astype(np.float32)

        for i in range(2):
            try:
                image = self.keys[index+3*i]
            except IndexError:
                # if the index is the last one
                images.append(images[0])
                continue
            image += ".png"
            margin_left, margin_top = 0, 0
            im = cv2.imread(self.folder + image)
            # scipy.misc.imsave('test'+str(index+3*i)+'.png', im)
            # Crop the image and normalize it
            if self.preprocess:
                margin_left, margin_top, _, _ = ROI
                im = preprocessImage(im, INPUT_WIDTH, INPUT_HEIGHT)
            images.append(im)
        # swap color axis
        images = np.dstack(images)
        # print(images.shape)
        self.numChannel = images.shape[2]
        images = images.transpose((2, 0, 1)).astype(np.float32)
        return th.from_numpy(images), th.from_numpy(y)

    def __len__(self):
        return len(self.keys)

class JsonDataset1(Dataset):
    def __init__(self, labels, folder="", preprocess=False, random_flip=0.0, swap=False):
        self.keys = list(labels.keys())
        self.labels = labels.copy()
        self.folder = folder
        self.preprocess = preprocess
        self.random_flip = random_flip
        self.swap = swap
        self.numChannel = 6

    def __getitem__(self, index):
        """
        :param index: (int)
        :return: (PyTorch Tensor, PyTorch Tensor)
        """
        images = []
        image = self.keys[index]
        labels = np.array(self.labels[image]).astype(np.float32)
        y = labels.flatten().astype(np.float32)

        for i in range(2):
            try:
                image = self.keys[index+i]
            except IndexError:
                # if the index is the last one
                images.append(images[0])
                continue
            image += ".png"
            margin_left, margin_top = 0, 0
            im = cv2.imread(self.folder + image)
            im = Image.fromarray(im)
            im = imgTransform(im)
            im = np.array(im)
            print('hhhhh')
            scipy.misc.imsave('test'+str(index+i)+'.png', im)
            # Crop the image and normalize it
            if self.preprocess:
                margin_left, margin_top, _, _ = ROI
                im = preprocessImage(im, INPUT_WIDTH, INPUT_HEIGHT)
            images.append(im)
        # swap color axis
        images = np.dstack(images)
        # print(images.shape)
        self.numChannel = images.shape[2]
        images = images.transpose((2, 0, 1)).astype(np.float32)
        return th.from_numpy(images), th.from_numpy(y)

    def __len__(self):
        return len(self.keys)

def computeMSE(y_test, y_true, indices):
    """
    Compute Mean Square Error
    and print its value for the different sets
    :param y_test: (numpy 1D array)
    :param y_true: (numpy 1D array)
    :param indices: [[int]] Indices of the different subsets
    """
    idx_train, idx_val, idx_test = indices
    # MSE Loss
    error = np.square(y_test - y_true)

    print('Train error={:.6f}'.format(np.mean(error[idx_train])))
    print('Val error={:.6f}'.format(np.mean(error[idx_val])))
    print('Test error={:.6f}'.format(np.mean(error[idx_test])))
    print('Total error={:.6f}'.format(np.mean(error)))
