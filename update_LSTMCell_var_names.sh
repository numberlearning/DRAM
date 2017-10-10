#!/bin/bash

# Due to Tensorflow update, the LSTM Cell weights and biases variable names have to be changed
# This code makes the changes to the last checkpoints in the 10 models
# The iteration of the checkpoint file that's being modified is the iteration listed in "checkpoint".

for i in {1..10}
do
    MODELNAME="DRAM_classify_blobs_2_run_${i}"

    iters=(0 250 1000 4000 16000 32000 64000 125000 250000)
    for j in "${iters[@]}"
    do
	sed -i '/model_checkpoint_path:/c\model_checkpoint_path: "classifymodel_'$j'.ckpt"' model_runs/$MODELNAME/checkpoint
	sed -i '/all_model_checkpoint_paths:/c\all_model_checkpoint_paths: "classifymodel_'$j'.ckpt"' model_runs/$MODELNAME/checkpoint
	cat model_runs/$MODELNAME/checkpoint
	python3 tensorflow_rename_variables.py --checkpoint_dir=model_runs/$MODELNAME --replace_from=biases --replace_to=bias
	python3 tensorflow_rename_variables.py --checkpoint_dir=model_runs/$MODELNAME --replace_from=weights --replace_to=kernel
    done
    sed -i '/model_checkpoint_path:/c\model_checkpoint_path: "classifymodel_2000000.ckpt"' model_runs/$MODELNAME/checkpoint
    sed -i '/all_model_checkpoint_paths:/c\all_model_checkpoint_paths: "classifymodel_2000000.ckpt"' model_runs/$MODELNAME/checkpoint
done

SOMEVAR='done running ^_^'  
echo "$SOMEVAR"
