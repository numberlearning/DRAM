#!/bin/bash
for i in {1..10}
do
    MODELNAME="estimation/classifier_model/classifier_DAA_const_fN_run${i}"
    echo $MODELNAME
    rsync -rP -e ssh "../../mtfang/DRAM/model_runs/$MODELNAME/" "sychen23@psy-pdp-guppy:DRAM/model_runs/$MODELNAME/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


