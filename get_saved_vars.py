from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
model_dir = "model_runs/DRAM_classify_blobs_2_run_1_test"
#model_dir = "model_runs/DRAM_classify_blobs_2_test"
for i in [0, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 125000, 500000]:
    checkpoint_path = os.path.join(model_dir, "classifymodel_" + str(i) + ".ckpt")
    #print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

    #Get just the names
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    #for key in var_to_shape_map:
        #print("tensor_name: ", key)
        #print(reader.get_tensor(key)) # Remove this is you want to print only variable names


    import collections
    ordered_map = collections.OrderedDict(sorted(var_to_shape_map.items()))
    for key in ordered_map.items():
        print(key)
