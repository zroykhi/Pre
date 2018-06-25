# Pre Project: ship's parameters prediction based on sea surface images
### Files explanations
`autoRun.sh`: runs all the codes automatically and generates the results. Usage: `bash autoRun.sh` or `sudo bash autoRun.sh` if your python modules is installed with sudo command.

`comparePare.xls`: a xls file which can calculate the parameters of a cnn network if you want to change some of parameters

`constants`: defines some main constants of the project

`models.py`: network models

`pltDiff.py`: used to plot figure comparing the original boat parameters and predictions ones

`pltModelTimegap.py`: read the result.txt and plot the model-timeGap-loss figures

`render.py`: used for render images data set with blender file. Run command:
```
blender model.blend --background --python render.py
```
or copy the code to blender's text editor and press "run script"

`result.txt`: contains the results of training (architectures of the network and test loss etc.). Generated automatically by `train.py`

`train.py`: used to train the model

`test.py`: used to predict the results

`transformData.py`: transform the data obtained from `blender.py` to the one which can be used for training

`insertKeyframe.py`: insert the prediction parameters into blender file `model.blend` and visualize the prediction in order to compare with the original ones. Usage: copy the code to blender's text editor and press "run script"

`model.blend`: 3d ocean-boat model, use to generate images dataset

`labels.json`: ground truth data of boat's parameters

### Step guidance:
1. download the code and add environment path by changing the code in
  + `transformData.py`
  + `render.py`
  + `autoRun.sh`

2. download the images dataset [here](https://drive.google.com/file/d/1Iez0P0yxOB1csu3ckNv2uvGewOaGSwr4/view?usp=sharing) if you have not yet and put the dataset under the directory **3dmodel**, make sure the dataset folder's name is **render619** and has file **labels.json** inside it

3. for train, goto Pre's parent folder and run command:
```
python3 -m Pre.train -f Pre/3dmodel/render619
```

4. for prediction, goto Pre's parent folder and run command:
```
python3 -m Pre.test -f Pre/3dmodel/render619
```

5. or you can run autoRun.sh directly
