"""
transform the data obtained from blender.py to the one which can be used for training
"""
import json
import argparse
import sys
# change your path here(path to 'Pre' folder)
# sys.path.append('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/Pre')
sys.path.append('/home/interns/Documents/ZHI/horizon-detection/Pre/')
from constants import TOTAL_FRAME, FRAME_INTERVAL
def transformDate(inputFile, output, TIME_GAP):
    labels = json.load(open(inputFile))
    print("transforming data ...")
    for i in range(len(labels)):
        if labels.get(str(i*FRAME_INTERVAL)) != None and i*FRAME_INTERVAL < (TOTAL_FRAME - TIME_GAP):
            labels[str(i*FRAME_INTERVAL)] = labels[str(i*FRAME_INTERVAL+TIME_GAP)]

    for i in range(TOTAL_FRAME - TIME_GAP, TOTAL_FRAME):
        if labels.get(str(i)) != None:
            labels.pop(str(i))

    for key, value in labels.items():
        # labels[key][2] = 90
        labels[key] = value[:-1]

    json.dump(labels,open(output,'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform parameters')
    parser.add_argument('-i', '--InputFile', help='Original json file', default="/render/paras_origin.json", type=str)
    parser.add_argument('-o', '--Output', help='Output file name', default='/render/labels.json', type=str)
    parser.add_argument('-t', '--TIME_GAP', help='Time gap', default=25, type=int)
    args = parser.parse_args()

    transformDate(inputFile=args.InputFile, output=args.Output, TIME_GAP=args.TIME_GAP)
