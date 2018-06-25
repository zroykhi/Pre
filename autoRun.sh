#!/bin/bash
# BASE_DIR=/home/interns/zhi
# BASE_DIR=/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection
BASE_DIR=/home/interns/Documents/ZHI/horizon-detection/
IMG_DIR=3dmodel/render619
CODE_DIR=/Pre/
IDX=1
RESULT_FILE=/result.txt
declare -a model_list=("cnn" "CNN_LSTM")
cd $BASE_DIR
start_tm=`date +%s%N`
for T in {20..30..1} ; do
    cd $BASE_DIR$CODE_DIR
    python3 3dmodel/transformData.py -i $IMG_DIR"/paras_origin.json" -o $IMG_DIR"/labels.json" -t ${T};
    cp $BASE_DIR$CODE_DIR$IMG_DIR"/labels.json" 3dmodel/labels_${T}.json
    cd $BASE_DIR
    for model in "${model_list[@]}"; do
        echo "loop No:"$IDX
        echo "current model:"${model}"; time gap: "$T
        python3 -m Pre.train -f $BASE_DIR$CODE_DIR$IMG_DIR --model_type ${model} -t ${T} --num_epochs 50
        python3 -m Pre.test -f $BASE_DIR$CODE_DIR$IMG_DIR -m ${model} -w ${model}"_model_"${T}"_tmp.pth" -t $T
        python $BASE_DIR$CODE_DIR"pltDiff.py" -m ${model} -t $T -o $BASE_DIR$CODE_DIR$IMG_DIR"/paras_origin.json" -p $BASE_DIR$CODE_DIR"3dmodel/predictions_"${model}"_"${T}".json"
    let IDX+=1
    echo -e "\n"
    done
done
cd $BASE_DIR$CODE_DIR
python pltModelTimegap.py -f $BASE_DIR$CODE_DIR$RESULT_FILE
end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{print($1-$2)/1000000000}'`
echo "time used:"$use_tm" s"
echo "All finished !"
