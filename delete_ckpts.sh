#!/bin/bash
for i in $(seq 1 10) 
do
    for iter in $(seq 1 10100000 10000)
    do
	if [ "$iter" -ne 0 -a  "$iter" -ne 100 -a "$iter" -ne 200 -a "$iter" -ne 300 -a "$iter" -ne 400 -a "$iter" -ne 600 -a "$iter" -ne 800 -a "$iter" -ne 1200 -a "$iter" -ne 1600 -a "$iter" -ne 2400 -a "$iter" -ne 3200 -a "$iter" -ne 4800 -a "$iter" -ne 6400 -a "$iter" -ne 9600 -a "$iter" -ne 12800 -a "$iter" -ne 19200 -a "$iter" -ne 25600 -a "$iter" -ne 38400 -a "$iter" -ne 51200 -a "$iter" -ne 76800 -a "$iter" -ne 102400 -a "$iter" -ne 153600 -a "$iter" -ne 204800 -a "$iter" -ne 307200 -a "$iter" -ne 409600 -a "$iter" -ne 614400 -a "$iter" -ne 819200 -a "$iter" -ne 1000000 -a "$iter" -ne 1228800 -a "$iter" -ne 1638400 -a "$iter" -ne 2000000 -a "$iter" -ne 2457600 -a "$iter" -ne 3000000 -a "$iter" -ne 3276800 -a "$iter" -ne 4000000 -a "$iter" -ne 4915200 -a "$iter" -ne 5000000 -a "$iter" -ne 6000000 -a "$iter" -ne 6553600 -a "$iter" -ne 7000000 ]
	then
	    MODELNAME="DRAM_twolayer_origin_run_${i}"
            rm model_runs/$MODELNAME/classifymodel_${iter}.ckpt.index
            rm model_runs/$MODELNAME/classifymodel_${iter}.ckpt.meta
            rm model_runs/$MODELNAME/classifymodel_${iter}.ckpt.data-00000-of-00001
        fi
    done
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


