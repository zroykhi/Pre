"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import
import json
import argparse
import time
from Pre.constants import INPUT_WIDTH, INPUT_HEIGHT, RES_DIR, ONE_IMG_ONLY, DATASET_SEQUENCE
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from Pre.utils import loadLabels, loadTestLabels, loadTrainLabels
# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
from Pre.models import ConvolutionalNetwork, CNN_RNN, CNN_LSTM
"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc
import os

evaluate_print = 1  # Print info every 1 epoch
VAL_BATCH_SIZE = 64  # Batch size for validation and test data

def main(test_folder, batchsize=8,
         seed=42, cuda=True, num_output=2, random_trans=0.4,
         model_type="CNN_LSTM", weights='', time_gap=25, output='none'):
    
    from Pre.utils import JsonDatasetOrigin as JsonDataset
    model = CNN_LSTM(num_channel=3)
    model.load_state_dict(th.load(weights))
    if not test_folder.endswith('/'):
        test_folder += '/'

    print('has cuda?', cuda)
    #images = [f for f in os.listdir(test_folder) if f.endswith('.png')]
    #images.sort(key=lambda name: int(name.split('.png')[0]))
    # loadtest lables function
    test_labels = json.load(open(test_folder + 'labels.json'))
    images = list(test_labels.keys())
    images.sort(key=lambda name: int(name.strip().replace('"',"")))
    # test_labels = {key: labels[key] for key in images}
    # print(test_labels)
    # print(images)
    #test_loader = th.utils.data.DataLoader(torch.from_numpy(X_test).long(), batch_size=batchsize) 
    #if val_folder == train_folder:
    #    train_labels, val_labels, test_labels, _ = loadLabels(train_folder, model_type)
    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)

    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    test_loader = th.utils.data.DataLoader(JsonDataset(test_labels, preprocess=True, folder=test_folder, random_trans=0, sequence=True),
                                           batch_size=batchsize, shuffle=False, **kwargs)

    # train_loader1 = th.utils.data.DataLoader(
    #    JsonDataset(train_labels, preprocess=True, folder='Pre/3dmodel/testData719/', random_trans=random_trans, sequence=DATASET_SEQUENCE),
    #    batch_size=batchsize, shuffle=True, **kwargs)

    #numChannel = train_loader.dataset.numChannel
    #if model_type == "cnn":
    #    model = ConvolutionalNetwork(num_channel=numChannel, num_output=num_output)
    #elif model_type == "CNN_RNN":
    #    model = CNN_RNN(num_channel=numChannel)
    #elif model_type == "CNN_LSTM":
    #    model = CNN_LSTM(num_channel=numChannel)
    #else:
    #    raise ValueError("Model type not supported")

    #if cuda:
    #    model.cuda()

    predictions = {}
    idx = 0
    print('start testing')
    for data in test_loader:
        #print('data type:', type(data))
        #print(data)
        #print(len(data))
        data = Variable(data[0])
        prediction = model(data)
        #print('prediction shape:', prediction.size)
        #print(prediction)
        #print(prediction.shape)
        row = prediction.shape[0]
        for i in range(row):
            print(prediction[i])
            ls = prediction[i].detach().numpy().tolist()
            #print(type(ls))
            #res = {images[idx]:ls}
            predictions.update({str(int(images[idx])+args.time_gap):ls})
            idx += 1
    jsObj = json.dumps(predictions)
    
    out = ''
    if args.output == 'none':
        out = './Pre/results/predictions_'+args.model+"_"+str(args.time_gap)+'.json'
    else:
        out = args.output
    fileObject = open(out, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('-f', '--test_folder', help='Test folder', type=str)
    parser.add_argument('-bs', '--batch_size', help='batch size', type=int, default=8)
    parser.add_argument('-w', '--weight', help='Model weight')
    parser.add_argument('-m', '--model', help='Model type', type=str, default='CNN_LSTM')
    parser.add_argument('-o', '--output', help='output file', default='none')
    parser.add_argument('-t', '--time_gap', help='time gap', default=25)
    args = parser.parse_args()

    main(test_folder=args.test_folder, batchsize=args.batch_size, weights=args.weight, model_type=args.model, output=args.output, time_gap=args.time_gap)

