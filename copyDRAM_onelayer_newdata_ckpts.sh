#!/bin/bash
for i in {1..10}
do
    MODELNAME="DRAM_onelayer_newdata4_run_${i}"
    echo $MODELNAME
    cd model_runs
    mkdir $MODELNAME
    cd ..
    rsync -rP -e ssh "../../mtfang/DRAM/model_runs/$MODELNAME/" "model_runs/$MODELNAME/"
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"


