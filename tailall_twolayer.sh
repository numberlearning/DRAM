#!/bin/bash

for i in $(seq 1 10)
do
    MODELNAME="DRAM_twolayer_filterimg_norm_run_${i}"
    tail -1 ./model_runs/$MODELNAME/nohup.out 
done

