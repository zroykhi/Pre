"""
this code is used to plot figure comparing the original boat parameters and predictions ones
"""
# run this code under ssh mode, you need to add the following two lines codes.
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.pyplot as plt
import json
import argparse
from constants import RES_DIR

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-m', '--model', help='Model type', default="cnn", type=str, choices=['cnn','CNN_RNN','CNN_LSTM'])
parser.add_argument('-t', '--time_gap', help='Time gap', default=25, type=int)
parser.add_argument('-o', '--origin_file', help='Original file', default="results/paras_origin.json", type=str)
parser.add_argument('-p', '--prediction_file', help='Predictin file', default="results/predictions.json", type=str)
args = parser.parse_args()

originData = args.origin_file
predData = args.prediction_file
labelsOrigin = json.load(open(originData))
labelsPred = json.load(open(predData))
frames1 = list(labelsOrigin.keys())
frames2 = list(labelsPred.keys())
# sort keys
frames1.sort(key=lambda name: int(name.strip().replace('"',"")))
frames2.sort(key=lambda name: int(name.strip().replace('"',"")))
# obtain x rotation and y rotation list respectively
originX = [labelsOrigin[str(key)][0] for key in frames1]
originY = [labelsOrigin[str(key)][1] for key in frames1]
predX = [labelsPred[str(key)][0] for key in frames2]
predY = [labelsPred[str(key)][1] for key in frames2]
# frames: str -> int
frames1 = [int(key) for key in frames1]
frames2 = [int(key) for key in frames2]
# plot roll
plt.figure(1)
plt.plot(frames1, originX, 'r-', label='original roll')
plt.plot(frames2, predX, 'b-', label='prediction roll')
plt.title("roll - frame")
plt.xlabel("Frames")
plt.ylabel("Roll")
plt.legend(loc='upper right')
plt.savefig(RES_DIR+args.model+'_'+str(args.time_gap)+'_roll'+'.png')
plt.show()

# plot pitch
plt.figure(2)
plt.plot(frames1, originY, 'r-', label='original pitch')
plt.plot(frames2, predY, 'b-', label='prediction pitch')
plt.title("pitch - frame")
plt.xlabel("Frames")
plt.ylabel("Pitch")
plt.legend(loc='upper right')
plt.savefig(RES_DIR+args.model+'_'+str(args.time_gap)+'_pitch'+'.png')
plt.show()
