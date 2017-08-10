#!/bin/bash
for i in {1..10}
do
    MODELNAME="DRAM_classify_blobs_run_${i}"
    mkdir "model_runs/$MODELNAME"
    sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
    nohup python3 -u DRAM_classify_blobs.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


