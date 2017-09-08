#!/bin/bash

# Due to Tensorflow update, the LSTM Cell weights and biases variable names have to be changed
# This code makes the changes to the last checkpoints in the 10 models

for i in {1..10}
do
    MODELNAME="DRAM_classify_blobs_2_run_${i}"
    python3 tensorflow_rename_variables.py --checkpoint_dir=model_runs/$MODELNAME --replace_from=biases --replace_to=bias
    python3 tensorflow_rename_variables.py --checkpoint_dir=model_runs/$MODELNAME --replace_from=weights --replace_to=kernel
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
