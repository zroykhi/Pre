#!/bin/bash
declare -a model_list=("cnn" "CNN_LSTM")
for T in {25..26} ; do 
    python 3dmodel/transformData.py -i 3dmodel/render619/paras_origin.json -o 3dmodel/render619/labels.json -t ${T}; 
    cp 3dmodel/render619/labels.json 3dmodel/labels_${T}.json
    cd ../
    for model in "${model_list[@]}"; do
        python3 -m Pre.train -f Pre/3dmodel/render619/ --model_type ${model} -t ${T} --num_epochs 1
        python3 -m Pre.test -f Pre/3dmodel/render619/ --model_type ${model} -w ${model}"_model_"${T}".pth"
#        python ./pltDiff.py
    cd Pre/
    done
done
