#!/bin/bash
for i in $(seq 4 4)
do
    MODELNAME="1task_0127_run_${i}"
    rsync -rP -e ssh "model_runs/$MODELNAME/" "mtfang@psy-pdp-hydra:DRAM/model_runs/$MODELNAME/"
    # rsync -rP -e ssh "model_runs/DRAM_twolayer_test_1015_2/" "mtfang@psy-pdp-guppy:DRAM/model_runs/DRAM_twolayer_test_1015_2/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


