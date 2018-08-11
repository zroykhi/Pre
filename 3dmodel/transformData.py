"""
transform the data obtained from blender.py to the one which can be used for training
"""
import json
import argparse
import sys
import os
# change your path here(path to 'Pre' folder)
# sys.path.append('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/Pre')
# sys.path.append('/home/interns/Documents/ZHI/horizon-detection/Pre/')
dirpath = os.getcwd()
sys.path.append(dirpath)
# sys.path.append('/home/theo/Pre/Code/horizon-detection/Pre/')
from constants import FRAME_INTERVAL
def transformDate(inputFile, output, TIME_GAP):
    labels = json.load(open(inputFile))
    TOTAL_FRAME = list(labels.keys())
    # TOTAL_FRAME = map(int, TOTAL_FRAME)
    TOTAL_FRAME = list(map(int, TOTAL_FRAME))
    MAX_FRAME = max(TOTAL_FRAME)
    MIN_FRAME = min(TOTAL_FRAME)
    print("transforming data ...")
    # add time gap to data
    # for i in range(len(labels)):
    for i in range(MIN_FRAME,MAX_FRAME+1):
        if labels.get(str(i*FRAME_INTERVAL)) != None and i*FRAME_INTERVAL < (MAX_FRAME - TIME_GAP):
            labels[str(i*FRAME_INTERVAL)] = labels[str(i*FRAME_INTERVAL+TIME_GAP)]
    # delete frames which exceed total frame
    for i in range(MAX_FRAME - TIME_GAP, MAX_FRAME):
        if labels.get(str(i)) != None:
            labels.pop(str(i))
    # use only pitch and roll to train, delete yaw
    for key, value in labels.items():
        labels[key] = value[:-1]

    json.dump(labels,open(output,'w'))
    print("transform data done !")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform parameters')
    parser.add_argument('-i', '--InputFile', help='Original json file', default="/render/paras_origin.json", type=str)
    parser.add_argument('-o', '--Output', help='Output file name', default='/render/labels.json', type=str)
    parser.add_argument('-t', '--TIME_GAP', help='Time gap', default=25, type=int)
    args = parser.parse_args()

    transformDate(inputFile=args.InputFile, output=args.Output, TIME_GAP=args.TIME_GAP)
