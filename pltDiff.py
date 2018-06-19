import matplotlib.pyplot as plt
import json
from constants import TIME_GAP

originData = "3dmodel/labels.json"
predData = "3dmodel/predictions.json"
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
frames1 = [int(key)+TIME_GAP for key in frames1]
frames2 = [int(key) for key in frames2]
# plot x rotation
plt.figure(1)
plt.plot(frames1, originX, 'r-', label='original x rotation')
plt.plot(frames2, predX, 'b-', label='prediction x rotation')
plt.title("X rotation - frame")
plt.xlabel("Frames")
plt.ylabel("X Rotations")
plt.legend(loc='upper right')
# plot y rotation
plt.figure(2)
plt.plot(frames1, originY, 'r-', label='original y rotation')
plt.plot(frames2, predY, 'b-', label='prediction y rotation')
plt.title("Y rotation - frame")
plt.xlabel("Frames")
plt.ylabel("Y Rotations")
plt.legend(loc='upper right')

plt.show()
