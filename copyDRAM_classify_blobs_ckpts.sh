#!/bin/bash
for i in {1..10}
do
    MODELNAME="DRAM_classify_blobs_2_run_${i}"
    rsync -rP -e ssh "model_runs/$MODELNAME/" "sychen23@psy-pdp-guppy:DRAM/model_runs/$MODELNAME/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


