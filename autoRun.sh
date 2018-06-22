#!/bin/bash
BASE_DIR=/home/interns/zhi
IMG_DIR=3dmodel/render619
CODE_DIR=/Pre/
IDX=1
RESULT_FILE=/result.txt
declare -a model_list=("cnn" "CNN_LSTM")
cd $BASE_DIR
for T in {25..26} ; do
    cd $BASE_DIR$CODE_DIR
    #python 3dmodel/transformData.py -i $IMG_DIR"/paras_origin.json" -o $IMG_DIR"/labels.json" -t ${T};
    cp $BASE_DIR$CODE_DIR$IMG_DIR"/labels.json" 3dmodel/labels_${T}.json
    cd $BASE_DIR
    for model in "${model_list[@]}"; do
        echo "loop No:"$IDX
        echo "current model:"${model}"; time gap: "$T
        python3 -m Pre.train -f $BASE_DIR$CODE_DIR$IMG_DIR --model_type ${model} -t ${T} --num_epochs 1
        python3 -m Pre.test -f $BASE_DIR$CODE_DIR$IMG_DIR -m ${model} -w ${model}"_model_"${T}"_tmp.pth"
        python $BASE_DIR$CODE_DIR"pltDiff.py" -m ${model} -t $T -o -p $BASE_DIR$CODE_DIR"3dmodel/predictions_"${model}"_"${T}".json"
    let IDX+=1
    echo -e "\n"
    done
done
cd $BASE_DIR$CODE_DIR
python pltModelTimegap.py -f $BASE_DIR$CODE_DIR$RESULT_FILE
echo "All finished !"
