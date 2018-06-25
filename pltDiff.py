import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from constants import TIME_GAP
import argparse

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-m', '--model', help='Model type', default="cnn", type=str, choices=['cnn','CNN_RNN','CNN_LSTM'])
parser.add_argument('-t', '--time_gap', help='Time gap', default=25, type=int)
parser.add_argument('-o', '--origin_file', help='Original file', default="3dmodel/paras_origin.json", type=str)
parser.add_argument('-p', '--prediction_file', help='Predictin file', default="3dmodel/predictions.json", type=str)
args = parser.parse_args()

originData = args.origin_file
predData = args.prediction_file
# predData = "3dmodel/predictions.json"
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
# plot x rotation
# plt.figure(1)
plt.plot(frames1, originX, 'r-', label='original x rotation')
plt.plot(frames2, predX, 'b-', label='prediction x rotation')
plt.title("X rotation - frame")
plt.xlabel("Frames")
plt.ylabel("X Rotations")
plt.legend(loc='upper right')
plt.savefig(args.model+'_'+str(args.time_gap)+'_x'+'.png')
# plt.show()
# plot y rotation
# plt.figure(2)
plt.plot(frames1, originY, 'r-', label='original y rotation')
plt.plot(frames2, predY, 'b-', label='prediction y rotation')
plt.title("Y rotation - frame")
plt.xlabel("Frames")
plt.ylabel("Y Rotations")
plt.legend(loc='upper right')
plt.savefig(args.model+'_'+str(args.time_gap)+'_y'+'.png')
# plt.show()
