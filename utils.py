from __future__ import print_function, division, absolute_import

import json
import re
import cv2
import numpy as np
import torch as th
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Pre.constants import MAX_WIDTH, MAX_HEIGHT, ROI, INPUT_HEIGHT, INPUT_WIDTH
from Pre.constants import SPLIT_SEED, TWO_IMG_GAP, LOADLABELS_SEQUENCE
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
    points = y.flatten()
    x = points[0]
    y = points[1]
    return x, y

# load train and test data from the same directory
def loadLabels(folder, modelType):
    if not folder.endswith('/'):
        folder += '/'
    labels = json.load(open(folder + 'labels.json'))
    images = list(labels.keys())
    images.sort(key=lambda name: int(name.strip().replace('"',"")))
    if LOADLABELS_SEQUENCE:
        train_keys = images[:int(round(0.6*len(images)))]
        val_keys = images[int(round(0.6*len(images))):int(round(0.8*len(images)))]
        test_keys = images[int(round(0.8*len(images))):len(images)]
    else:
        train_keys, tmp_keys = train_test_split(list(labels.keys()), test_size=0.4, random_state=SPLIT_SEED)
        val_keys, test_keys = train_test_split(tmp_keys, test_size=0.5, random_state=SPLIT_SEED)

    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}
    test_labels = {key: labels[key] for key in test_keys}

    print("{} images".format(len(labels)))
    return train_labels, val_labels, test_labels, labels

# load train data and test data from different directories
def loadTrainLabels(folder, modelType):
    if not folder.endswith('/'):
        folder += '/'
    labels = json.load(open(folder + 'labels.json'))
    images = list(labels.keys())
    images.sort(key=lambda name: int(name.strip().replace('"',"")))
    train_keys = images
    train_labels = {key: labels[key] for key in train_keys}

    print("train {} images".format(len(labels)))
    return train_labels, labels

# load train data and test/validation data from different directories
def loadTestLabels(folder, modelType):
    if not folder.endswith('/'):
        folder += '/'
    labels = json.load(open(folder + 'labels.json'))
    images = list(labels.keys())
    images.sort(key=lambda name: int(name.strip().replace('"',"")))
    if LOADLABELS_SEQUENCE:
        images.sort(key=lambda name: int(name.strip().replace('"',"")))
        val_keys = images[:int(round(0.5*len(images)))]
        test_keys = images[int(round(0.5*len(images))):len(images)]
    else:
        val_keys, test_keys = train_test_split(list(labels.keys()), test_size=0.5, random_state=SPLIT_SEED)

    val_labels = {key: labels[key] for key in val_keys}
    test_labels = {key: labels[key] for key in test_keys}

    print("test and val {} images in total".format(len(labels)))
    return val_labels, test_labels, labels

"""
take one image only as input
"""
class JsonDatasetOne(Dataset):
    def __init__(self, labels, folder="", preprocess=False, random_trans=0.0, swap=False, sequence=False):
        self.sequence = sequence
        self.labels = labels.copy()
        self.folder = folder
        self.preprocess = preprocess
        self.random_trans = random_trans
        self.swap = swap
        self.numChannel = 3
        self.keys = list(labels.keys())
        if self.sequence:
            self.keys.sort(key=lambda name: int(name.strip().replace('"',"")))

    def __getitem__(self, index):
        """
        :param index: (int)
        :return: (PyTorch Tensor, PyTorch Tensor)
        """
        image = self.keys[index]
        image += ".png"
        # im = cv2.imread(self.folder + image)
        im = Image.open(self.folder + image).convert("RGB") # RGB(270, 486, 3)
        im = np.asarray(im)
        # im.flags.writeable = True
        # im[...,[0,2]] = im[...,[2,0]] # RGB-> BGR
        if np.random.random() < self.random_trans:
            im = Image.fromarray(im)
            im = imgTransform(im)
            im = np.array(im)
        # Crop the image and normalize it
        if self.preprocess:
            im = preprocessImage(im, INPUT_WIDTH, INPUT_HEIGHT)
        image = image.replace(".png","")
        labels = np.array(self.labels[image]).astype(np.float32)
        y = labels.flatten().astype(np.float32)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1)).astype(np.float32)
        return th.from_numpy(im), th.from_numpy(y)

    def __len__(self):
        return len(self.keys)

"""
take two images as input and use data augmentation
"""
class JsonDatasetTwo(Dataset):
    def __init__(self, labels, folder="", preprocess=False, random_trans=0.0, swap=False, sequence=False):
        self.keys = list(labels.keys())
        self.sequence = sequence
        self.labels = labels.copy()
        self.folder = folder
        self.preprocess = preprocess
        self.random_trans = random_trans
        self.swap = swap
        self.numChannel = 6
        if self.sequence:
            self.keys.sort(key=lambda name: int(name.strip().replace('"',"")))

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
                image = self.keys[index+TWO_IMG_GAP*i]
            except IndexError:
                # if the index is the last one
                images.append(images[0])
                continue
            image += ".png"
            margin_left, margin_top = 0, 0
            # im = cv2.imread(self.folder + image) # BGR(270, 486, 3)
            im = Image.open(self.folder + image).convert("RGB") # RGB(270, 486, 3)
            # print(im.mode)
            im = np.asarray(im)
            # im.flags.writeable = True
            # im[...,[0,2]] = im[...,[2,0]] # RGB-> BGR
            # scipy.misc.imsave('before'+str(index)+'.png', im)
            if np.random.random() < self.random_trans:
                im = Image.fromarray(im)
                im = imgTransform(im)
                im = np.array(im)
                # scipy.misc.imsave('after'+str(index)+'.png', im)
            # Crop the image and normalize it
            if self.preprocess:
                margin_left, margin_top, _, _ = ROI
                im = preprocessImage(im, INPUT_WIDTH, INPUT_HEIGHT)
            images.append(im)
        images = np.dstack(images)
        # print(images.shape) # (90, 162, 6)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
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
