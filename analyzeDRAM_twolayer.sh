#!/bin/bash

MODELNAME="3task_half"
mkdir "model_runs/$MODELNAME"
mkdir "data/$MODELNAME"
sed -i '/model_name/c\model_name = "'$MODELNAME'"' model_settings.py
nohup python3 -u analyze_3task_half.py $MODELNAME > "model_runs/$MODELNAME/nohup.out" &

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
