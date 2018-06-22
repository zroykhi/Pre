from __future__ import print_function, division, absolute_import

import json
import re
import cv2
import numpy as np
import torch as th
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from constants import MAX_WIDTH, MAX_HEIGHT, ROI, INPUT_HEIGHT, INPUT_WIDTH, SPLIT_SEED


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

    images = list(labels.keys())
    images.sort(key=lambda name: int(name.strip().replace('"',"")))
    # print(images)
    # Split the data into three subsets
    if False:# modelType == "CNN_RNN":
        train_keys = images[:int(round(0.5*len(images)))]
        val_keys = images[int(round(0.5*len(images))):int(round(0.75*len(images)))]
        test_keys = images[int(round(0.75*len(images))):len(images)]
    else:
        train_keys, tmp_keys = train_test_split(list(labels.keys()), test_size=0.4, random_state=SPLIT_SEED)
        val_keys, test_keys = train_test_split(tmp_keys, test_size=0.5, random_state=SPLIT_SEED)

    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}
    test_labels = {key: labels[key] for key in test_keys}

    print("{} images".format(len(labels)))
    return train_labels, val_labels, test_labels, labels


class JsonDataset(Dataset):
    def __init__(self, labels, folder="", preprocess=False, random_flip=0.0, swap=False):
        self.keys = list(labels.keys())
        self.labels = labels.copy()
        self.folder = folder
        self.preprocess = preprocess
        self.random_flip = random_flip
        self.swap = swap

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

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1)).astype(np.float32)
        return th.from_numpy(im), th.from_numpy(y)

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
