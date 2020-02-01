#!/bin/bash

for i in $(seq 1 5)
do
    MODELNAME="DAA_NEW_TWOHIDDENS_30_APR24TH${i}"
    mkdir "model_runs/$MODELNAME"
    sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
    nohup python3 -u FF_estimation2.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &
done 

SOMEVAR='done running ^_^'
echo "$SOMEVAR"
