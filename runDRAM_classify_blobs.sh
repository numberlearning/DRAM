#!/bin/bash
for i in {1..10}
do
    sed -i '$d' model_settings.py
    MODELNAME="DRAM_classify_blobs_run_${i}"
    echo "model_name = '$MODELNAME'" >> model_settings.py

    #mkdir "model_runs/$MODELNAME"
    #nohup python3 -u DRAM_classify_blobs.py > "model_runs/$MODELNAME/nohup.out&"
done

ls -lah  
SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


