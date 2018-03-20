#!/bin/bash

for i in $(seq 1 3)
do
    MODELNAME="3task_simp_norm_run_${i}"
    mkdir "model_runs/$MODELNAME"
    sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
    nohup python3 -u Ct_2l_3t_TT_simp.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
