#!/bin/bash
for i in {1..10}
do
    MODELNAME="DAA_decrs_fN_5layer_run${i}"
    echo $MODELNAME
    rsync -rP -e ssh "../../mtfang/DRAM/model_runs/$MODELNAME/" "sychen23@psy-pdp-guppy:DRAM/model_runs/$MODELNAME/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


