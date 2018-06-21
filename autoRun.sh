#!/bin/bash
declare -a model_list=("cnn" "CNN_LSTM")
for T in {5..7} ; do 
    python 3dmodel/transformData.py -i 3dmodel/render619/paras_origin.json -o 3dmodel/render619/labels.json -t ${T}; 
    cd ../
    for model in "${model_list[@]}; do
        python3 -m Pre.train -f Pre/3dmodel/render619/ --model_type ${model}
    done
done
