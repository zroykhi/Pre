import re
import matplotlib.pyplot as plt
import argparse

def getList(filename):
    model = []
    trainError = []
    valError = []
    testError = []
    timeGap = []
    f = open(filename)
    lines = f.readlines()
    # with open(filename, 'r') as f:
    for line in lines:
#        while True:
  #      line = f.readline()
        if not line:
            continue
        line = line.strip('\n')
        line = line.replace(':',' ')
        line = re.split(" ", line)
        # print(line)
        if len(line)<2:
            continue
        if line[1] == "model":
         #   line[2] = line[2].replace('\n',' ')
            model.append(line[2])
        elif line[1] == "train":
           # line = line[3]#.split(":").split("(")
            trainError.append(float(line[3]))
        elif line[1] == "validation":
           # line = line[3]#.split(":").split("(")
            valError.append(float(line[3]))
        elif line[1] == "test":
           # line = line[3].split(":").split("(")
            testError.append(float(line[3]))
        elif line[1] == "gap":
           # line = line.split(":")
            timeGap.append(int(line[3]))
        else:
            # print("Error !")
            pass
        print(line)
    #print(model)      
    f.close()
    return model, trainError, valError, testError, timeGap

def find_repeat(source, elmt):
    elmt_index = []
    s_idx, e_idx = 0, len(source)
    while (s_idx < e_idx):
        try:
            temp = source.index(elmt, s_idx, e_idx)
            elmt_index.append(temp)
            s_idx = temp + 1
        except ValueError:
            break
    return elmt_index

def pltFigure(modelname, matrix, idx):
    plt.figure(idx)
    styles = ['r-', 'b-', 'g-', 'p-']
    labels = ['train loss', 'test loss', 'val loss']
    for i in range(len(matrix)-1):
        plt.plot(matrix[0], matrix[i+1], styles[i], label='train loss')
        plt.plot(matrix[0], predX, 'b-', label='prediction x rotation')
        plt.title(modelname + " - timeGap - loss")
        plt.xlabel("Time gap")
        plt.ylabel("loss")
    plt.legend(loc='upper right')
    plt.savefig(modelname+'_'+'loss_comparation'+'.png')
    # plt.show()

def main(filename):
    models, trainErrors, valErrors, testErrors, timeGaps = getList(filename)
    unqModels = set(models)
    posIdxMatrix = []
    trainLossMatrix = []
    testLossMatrix = []
    valLossMatrix = []
    timeGapMatrix = []
    for model in unqModels:
        posIdxMatrix.append(find_repeat(models, model))
    unqModels = list(unqModels)
    for i in range(len(unqModels)):
        resMatrix = []
        # resMatrix.append(posIdxMatrix[i])
        tmp = [timeGaps[idx] for idx in posIdxMatrix[i]]
        resMatrix.append(tmp)
        tmp = [trainErrors[idx] for idx in posIdxMatrix[i]]
        resMatrix.append(tmp)
        tmp = [testErrors[idx] for idx in posIdxMatrix[i]]
        resMatrix.append(tmp)
        tmp = [valErrors[idx] for idx in posIdxMatrix[i]]
        resMatrix.append(tmp)
        pltFigure(unqModels[i], resMatrix, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a line detector')
    parser.add_argument('-f', '--file', help='Input file', default="result.txt", type=str)
    args = parser.parse_args()
    print("start ploting ...")
    main(args.file)
    print("finished ploting !")
