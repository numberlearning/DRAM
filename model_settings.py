#e.g. 5, 5, 0, 9
#1, 1, 1, 9 or 15 for load_teacher
#7, 7, 1, 5 for 20x400 images, sensical
# 9 for trace_data
# 15 for trace_data_new
min_edge = 2
max_edge = 5
min_blobs = 1
max_blobs = 9

learning_rate = 1e-2

#11 when running DRAMcopy14, load_teacher, trace_data
#5, etc. when running DRAMcopy13, load_input, create_data
glimpses = max_blobs + 1#11

#1 when running viz_count and retina filterbank
#77 when running DRAMcopy14 and move_attn
#100 when running DRAMcopy13 (and 1 for viz?)
#10000 when running update_curves, classify_imgs2, match num_imgs
<<<<<<< HEAD
batch_size = 1

#change this whenever DRAM is run (before running nohup, make sure to make a directory in model_runs)
model_name = "rewrite_filterbank_Retina_test001"
=======
batch_size = 77

#change this whenever DRAM is run (before running nohup, make sure to make a directory in model_runs)
model_name = "DRAM_move_attn_sigmoid/run_5"
>>>>>>> 203124eb08ce289576d48fa05c7e25179a84620e
