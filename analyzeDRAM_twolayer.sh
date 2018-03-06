#!/bin/bash

MODELNAME="3task_prep_0216"
mkdir "model_runs/$MODELNAME"
mkdir "data/$MODELNAME"
sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
nohup python3 -u analyze_prep_3task.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
