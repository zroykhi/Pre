#!/bin/bash
BASE_DIR=/home/interns/zhi
IMG_DIR=/3dmodel/render619
CODE_DIR=/Pre
IDX=1
declare -a model_list=("cnn" "CNN_LSTM")
cd $BASE_DIR
for T in {25..26} ; do 
    #python 3dmodel/transformData.py -i 3dmodel/render619/paras_origin.json -o 3dmodel/render619/labels.json -t ${T}; 
    #cp 3dmodel/render619/labels.json 3dmodel/labels_${T}.json
    cd $BASE_DIR
    for model in "${model_list[@]}"; do
        echo "loop No:"$IDX
        echo "current model:"${model_list[@]}"; time gap: "$T
        python3 -m Pre.train -f $BASE_DIR$CODE_DIR$IMG_DIR --model_type ${model} -t ${T} --num_epochs 1
        python3 -m Pre.test -f $BASE_DIR$CODE_DIR$IMG_DIR -m ${model} -w ${model}"_model_"${T}"_tmp.pth"
#        python ./pltDiff.py
    cd $BASE_DIR
    let IDX+=1
    echo -e "\n"
    done
done
