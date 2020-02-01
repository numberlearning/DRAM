#!/bin/bash

for i in $(seq 1 3)
do
    MODELNAME="FF_LearningRate_${i}"
    mkdir "model_runs/$MODELNAME"
    sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
    nohup python3 -u FF_estimation.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &
done 

SOMEVAR='done running ^_^'
echo "$SOMEVAR"
