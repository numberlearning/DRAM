from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
model_dir = "model_runs/positions"
checkpoint_path = os.path.join(model_dir, "classifymodel_5000.ckpt")
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)
