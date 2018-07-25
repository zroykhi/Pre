"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
from Pre.constants import INPUT_WIDTH, INPUT_HEIGHT, RES_DIR
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from Pre.utils import JsonDataset, loadLabels, JsonDataset1
from Pre.models import ConvolutionalNetwork, CNN_RNN, CNN_LSTM
# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
"""if above line didnot work, use following two lines instead"""
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc

evaluate_print = 1  # Print info every 1 epoch
VAL_BATCH_SIZE = 64  # Batch size for validation and test data

def main(folder, num_epochs=100, batchsize=16,
         learning_rate=0.0001, seed=42, cuda=True, num_output=2, random_flip=0.5,
         model_type="cnn", evaluate_print=1, load_model="", time_gap=25):

    if not folder.endswith('/'):
        folder += '/'

    train_labels, val_labels, test_labels, _ = loadLabels(folder, model_type)

    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)

    # Retrieve number of samples per set
    n_train, n_val, n_test = len(train_labels), len(val_labels), len(test_labels)

    # Keywords for pytorch dataloader
    kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}
    # Create data loaders
    train_loader = th.utils.data.DataLoader(
        JsonDataset(train_labels, preprocess=True, folder=folder, random_flip=random_flip),
        batch_size=batchsize, shuffle=True, **kwargs)

    # Random flip also for val ?
    val_loader = th.utils.data.DataLoader(JsonDataset(val_labels, preprocess=True, folder=folder),
                                          batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    test_loader = th.utils.data.DataLoader(JsonDataset(test_labels, preprocess=True, folder=folder),
                                           batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    train_loader1 = th.utils.data.DataLoader(
        JsonDataset1(train_labels, preprocess=True, folder=folder, random_flip=random_flip),
        batch_size=batchsize, shuffle=True, **kwargs)

    # Random flip also for val ?
    val_loader1 = th.utils.data.DataLoader(JsonDataset1(val_labels, preprocess=True, folder=folder),
                                          batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    test_loader1 = th.utils.data.DataLoader(JsonDataset1(test_labels, preprocess=True, folder=folder),
                                           batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    numChannel = train_loader1.dataset.numChannel
    if model_type == "cnn":
        model = ConvolutionalNetwork(num_channel=numChannel, num_output=num_output, drop_p=0.0)
    elif model_type == "CNN_RNN":
        model = CNN_RNN(num_channel=numChannel)
    elif model_type == "CNN_LSTM":
        model = CNN_LSTM(num_channel=numChannel)
    else:
        raise ValueError("Model type not supported")

    if cuda:
        model.cuda()

    # L2 penalty
    weight_decay = 1e-4
    #weight_decay = 0
    # Optimizers
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay, nesterov=True)

    # Loss functions
    loss_fn = nn.MSELoss(size_average=False)
    # loss_fn = nn.SmoothL1Loss(size_average=False)
    best_error = np.inf
    best_train_error = np.inf
    train_err_list = []
    val_err_list = []
    xdata = []
    model_name = "{}_model_{}_tmp".format(model_type, time_gap)
    best_model_path = "{}.pth".format(model_name)
    best_model_path = RES_DIR + best_model_path
    fig = plt.figure()
    ax = fig.add_subplot(111)
    li, = ax.plot(xdata, train_err_list, 'b-', label='train loss')
    l2, = ax.plot(xdata, val_err_list, 'r-', label='val loss')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    plt.title("train epochs - loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show(block=False)
    # Finally, launch the training loop.
    start_time = time.time()
    print("Starting training...")
    # We iterate over epochs:
    for epoch in tqdm(range(num_epochs)):
        # Switch to training mode
        model.train()
        train_loss, val_loss = 0.0, 0.0
        # Full pass on training data
        # Update the model after each minibatch
        # for i, (inputs, targets) in enumerate(train_loader1):
        #     if cuda:
        #         inputs, targets = inputs.cuda(), targets.cuda()
        #     inputs, targets = Variable(inputs), Variable(targets)
        #     optimizer.zero_grad()
        #     predictions = model(inputs)
        #     loss = loss_fn(predictions, targets)
        #     loss.backward()
        #     # train_loss += loss.item()
        #     optimizer.step()

        for i, (inputs, targets) in enumerate(train_loader):
            # Adjust learning rate
            # adjustLearningRate(optimizer, epoch, num_epochs, lr_init=learning_rate,
            #                         batch=i, n_batch=len(train_loader), method='multistep')
            # Move variables to gpu
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            # print(inputs.shape) # [4, 3, 75, 100]
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Do a full pass on validation data
        model.eval()
        for inputs, targets in val_loader:
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Set volatile to True because we don't need to compute gradient
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

        # Compute error per sample
        val_error = val_loss / n_val
        if val_error < best_error:
            best_error = val_error
            # Move back weights to cpu
            if cuda:
                model.cpu()
            # Save Weights
            th.save(model.state_dict(), best_model_path)
            if cuda:
                model.cuda()
        if (train_loss / n_train) < best_train_error:
            best_train_error = train_loss / n_train

        if (epoch + 1) % evaluate_print == 0:
            # Then we print the results for this epoch:
            # Losses are averaged over the samples
            train_l = train_loss / n_train
            xdata.append(epoch+1)
            train_err_list.append(train_l)
            val_err_list.append(val_error)
            li.set_xdata(xdata)
            li.set_ydata(train_err_list)
            l2.set_xdata(xdata)
            l2.set_ydata(val_err_list)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            time.sleep(0.01)
            fig.show()
            # print("Epoch {} of {} took {:.3f}s".format(
            #     epoch + 1, num_epochs, time.time() - start_time))
            # print("  training loss:\t\t{:.6f}".format(train_loss / n_train))
            # print("  validation loss:\t\t{:.6f}".format(val_error))
    plt.savefig(RES_DIR+args.model_type+'_'+str(args.time_gap)+'_loss'+'.png')
    # After training, we compute and print the test error:
    model.load_state_dict(th.load(best_model_path))
    test_loss = 0.0
    for inputs, targets in test_loader:
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        test_loss += loss.item()
    print("Final results:")
    print("  best validation loss:\t\t{:.6f}".format(best_error))
    print("  test loss:\t\t\t{:.6f}".format(test_loss / n_test))
    # write result into result.txt
    # format fixed because we use this file later in pltModelTimegap.py
    with open("./Pre/result.txt", "a") as f:
        f.write("current model: ")
        f.write(model_type)
        f.write("\nbest train error:")
        f.write(str(best_train_error))
        f.write("\nbest validation loss:")
        f.write(str(best_error))
        f.write("\nfinal test loss:")
        f.write(str(test_loss / n_test))
        f.write("\n")
        f.write("time gap is:")
        f.write(str(time_gap))
        f.write("\n")
        f.write(str(model))
        f.write("\n\n")
    f.close()
    print("Total train time: {:.2f} mins".format((time.time() - start_time)/60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default=30, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=4, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training')
    parser.add_argument('--load_model', help='Start from a saved model', default="", type=str)
    parser.add_argument('--model_type', help='Model type: cnn', default="cnn", type=str, choices=['cnn','CNN_RNN','CNN_LSTM'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-t', '--time_gap', help='Time gap', default=25, type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()
    main(folder=args.folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
         learning_rate=args.learning_rate, cuda=args.cuda,
         seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=args.time_gap)
