#!/bin/bash
for i in $(seq 10 10)
do
    MODELNAME="DRAM_twolayer_filterimg_norm_run_${i}"
    rsync -rP -e ssh "model_runs/$MODELNAME/" "mtfang@psy-pdp-guppy:DRAM/model_runs/$MODELNAME/"
    # rsync -rP -e ssh "model_runs/DRAM_twolayer_test_1015_2/" "mtfang@psy-pdp-guppy:DRAM/model_runs/DRAM_twolayer_test_1015_2/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


