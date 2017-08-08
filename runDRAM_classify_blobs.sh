#!/bin/bash
for i in {1..10}
do
    mkdir "model_runs/DRAM_classify_blobs_run_${i}"
    nohup python3 -u DRAM_classify_blobs.py > "model_runs/DRAM_classify_blobs_run_${i}/nohup.out&"
done

ls -lah  
SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


