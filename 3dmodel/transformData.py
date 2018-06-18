import json
from ..constants import TIME_GAP, TOTAL_FRAME, FRAME_INTERVAL
def transformDate(input, output):
    labels = json.load(open(input))
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
    parser.add_argument('-i', '--Input', help='Original json file', default="/render/paras_origin.json", type=str)
    parser.add_argument('-o', '--Output', help='Output file name', default='/render/labels.json', type=str)
    args = parser.parse_args()

    transformDate(Input=args.Input, Output=args.Output)
