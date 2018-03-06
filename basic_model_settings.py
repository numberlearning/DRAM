#e.g. 5, 5, 0, 9
#1, 1, 1, 9 or 15 for load_teacher
#7, 7, 1, 5 for 20x400 images, sensical
# 9 for trace_data
# 15 for trace_data_new and DRAM_classify_blobs
min_edge = 2
max_edge = 5
min_blobs_train = 1
max_blobs_train = 15
min_blobs_test = 1
max_blobs_test = 15
hidden_size = 100
learning_rate = 0 # JLM: use default if 0

#11 when running DRAMcopy14, load_teacher, trace_data
#5, etc. when running DRAMcopy13, load_input, create_data
glimpses = 3

#1 when running viz_count
#77 when running DRAMcopy14
#100 when running DRAMcopy13 (and 1 for viz?) and DRAM_classify_blobs
#10000 (or 9000) when running update_curves, classify_imgs2, match num_imgs
batch_size = 100

#change this whenever DRAM is run (before running nohup, make sure to make a directory in model_runs)
model_name = "BASIC_run_10"