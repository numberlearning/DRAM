#!/bin/bash
for i in {1..10}
do
    MODELNAME="DRAM_classify_blobs_4_run_${i}"
    echo $MODELNAME
    rsync -rP -e ssh "model_runs/$MODELNAME/" "sychen23@psy-pdp-guppy:DRAM/model_runs/$MODELNAME/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


