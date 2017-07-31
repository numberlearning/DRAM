#e.g. 5, 5, 0, 9
#1, 1, 1, 9 for load_teacher
min_edge = 5
max_edge = 5
min_blobs = 0
max_blobs = 9

learning_rate = 1e-3

#11 when running DRAMcopy14, load_teacher, trace_data
#5, etc. when running DRAMcopy13, load_input, create_data
glimpses = 10#11

#1 when running viz_count
#77 when running DRAMcopy14
#100 when running DRAMcopy13 (and 1 for viz?)
#10000 when running update_curves, classify_imgs2, match num_imgs
batch_size = 100 #77#100#77

#change this whenever DRAM is run (before running nohup, make sure to make a directory in model_runs)
model_name = "rewrite_filterbank"
