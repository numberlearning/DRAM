#!/bin/bash

#sed -i '/min_edge/c\min_edge = 2' model_settings.py
#sed -i '/max_edge/c\max_edge = 5' model_settings.py
#sed -i '/min_blobs/c\min_blobs = 1' model_settings.py
#sed -i '/max_blobs/c\max_blobs = 9' model_settings.py

for i in {1..10}
do
    MODELNAME="DRAM_classify_blobs_3_run_${i}"
    mkdir "model_runs/$MODELNAME"
    sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
    nohup python3 -u DRAM_classify_blobs.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
