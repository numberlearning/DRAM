#e.g. 5, 5, 0, 9
#1, 1, 1, 9 for load_teacher
min_edge = 1
max_edge = 1
min_blobs = 1
max_blobs = 9

learning_rate = 1e-3

#11 when running DRAMcopy14, load_teacher, trace_data
#5, etc. when running DRAMcopy13, load_input, create_data
glimpses = max_blobs + 1#11

#1 when running viz_count
#77 when running DRAMcopy14
#100 when running DRAMcopy13 (and 1 for viz?)
#10000 when running update_curves, classify_imgs2, match num_imgs
batch_size = 77#77#100#77

#change this whenever DRAM is run (before running nohup, make sure to make a directory in model_runs)
model_name = "attn_window_positions"
