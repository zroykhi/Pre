import json
from constants import TIME_GAP, TOTAL_FRAME, FRAME_INTERVAL

labels = json.load(open('paras.json'))
for i in range(len(labels)):
    if labels.get(str(i*FRAME_INTERVAL)) != None and i*FRAME_INTERVAL < (TOTAL_FRAME - TIME_GAP):
        labels[str(i*FRAME_INTERVAL)] = labels[str(i*FRAME_INTERVAL+TIME_GAP)]

for i in range(TOTAL_FRAME - TIME_GAP, TOTAL_FRAME):
    if labels.get(str(i)) != None:
        labels.pop(str(i))

for key, value in labels.items():
    # labels[key][2] = 90
    labels[key] = value[:-1]
    
json.dump(labels,open('labels.json','w'))
