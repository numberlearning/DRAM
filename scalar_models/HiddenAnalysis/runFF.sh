#!/bin/bash

for i in $(seq 1 10)
do
    MODELNAME="New_CAA_decrs_fN_scalar${i}"
    mkdir "model_runs/$MODELNAME"
    sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
    nohup python3 -u FF_estimation_scalar.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &
done 

SOMEVAR='done running ^_^'
echo "$SOMEVAR"
