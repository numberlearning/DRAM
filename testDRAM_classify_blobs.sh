#!/bin/bash

#MODELNAME="DRAM_classify_blobs_2_test"
MODELNAME="DRAM_classify_blobs_2_run_1_test"
mkdir "model_runs/$MODELNAME"
sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
python3 DRAM_classify_blobs.py $MODELNAME

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
