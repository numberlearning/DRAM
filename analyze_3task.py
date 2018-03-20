import numpy as np
from analysis_count_0g import read_n, classify_imgs2
from model_settings import test_trials
import operator

model_name = "3task_half"
num_runs = 3
iter_list = np.arange(0,61000,1000)
glimpse_list = np.arange(0,10,1)
max_blobs = 9
min_blobs = 1

num_iters = len(iter_list)
num_glimpses = len(glimpse_list)
blob_list = np.arange(0,10,1)
output_size = max_blobs - min_blobs + 2 # 10
data_directory = "data/" + model_name + "/"
m = 0.5
num_imgs = test_trials*(output_size-1)

confidence_all_runs = np.zeros([num_runs, num_iters, output_size-1, num_glimpses, 1, output_size]) 
choice_all_runs = np.zeros([num_runs, num_iters, output_size-1, num_glimpses, 1, output_size])
countword_all_runs = np.zeros([num_runs, num_iters, num_imgs, output_size+1])
pointblob_all_runs = np.zeros([num_runs, num_iters, num_imgs, output_size+1])
point_target_all_runs = np.zeros([num_runs, num_iters, num_imgs, num_glimpses])
blob_point_all_runs = np.zeros([num_runs, num_iters, num_imgs, output_size+1])
class_all_runs = np.zeros([num_runs, num_iters, num_imgs])
point_results_all_runs = np.zeros([num_runs, num_iters, num_imgs, num_glimpses])

def fill_matrix(path, iteration):
    """Fill the confidence and choice matrices for one run at one iteration."""
            
    data = None
    num_imgs = test_trials*(output_size-1) # batch_size
    imgs_data = classify_imgs2(iteration, True, num_imgs, path=path) # new_imgs = True
    confidence_one_run = np.zeros([output_size-1, num_glimpses, 1, output_size])
    choice_one_run = np.zeros([output_size-1, num_glimpses, 1, output_size])
    countword_one_run = np.zeros([num_imgs, output_size+1])
    pointblob_one_run = np.zeros([num_imgs, output_size+1])
    point_target_one_run = np.zeros([num_imgs, num_glimpses])
    blob_point_one_run = np.zeros([num_imgs, output_size+1])
    class_one_run = np.zeros(num_imgs)
    point_results_one_run = np.zeros([num_imgs, num_glimpses]) # 1 to 9

    for nb in range(output_size-1):
        confidence_hist = np.zeros([num_glimpses, 1, output_size])
        choice_hist = np.zeros([num_glimpses, 1, output_size])
        num_imgs_with_num_blobs = 0.00001

        for idx, data in enumerate(imgs_data):
            if data["label"][nb] == 1: # data is for an image with nb+1 blobs
                num_imgs_with_num_blobs += 1
                countword_one_run[idx][0]=nb+1
                pointblob_one_run[idx][0]=nb+1

                for g, glimpse in enumerate(glimpse_list):
                    # Histogram of softmaxes
                    confidence_hist[glimpse] += data["classifications"][glimpse]
                    # Histogram of choices
                    choice = np.argmax(data["classifications"][glimpse])
                    choice_list = [0] * (output_size)
                    choice_list[choice] = 1
                    choice_hist[glimpse] += choice_list
                    # Count Word
                    countword_one_run[idx][glimpse+1]=choice
                    # Point Blob
                    pointblob_one_run[idx][glimpse+1]=int(data["corrects"][glimpse])

                pointblob_one_run[idx][nb+2]=nb+1
                point_target_one_run[idx] = data["corrects"]
                point_target_one_run[idx][nb+1] = nb+1
                class_one_run[idx] = data["class"]+1
        confidence_hist = confidence_hist / num_imgs_with_num_blobs
        confidence_one_run[nb] = confidence_hist.tolist()
        choice_hist = choice_hist / num_imgs_with_num_blobs
        choice_one_run[nb] = choice_hist.tolist()

    for idx, data in enumerate(imgs_data):
        for g in range(1,num_glimpses+1):
            blob_point_one_run[idx][g] = int(data["blob_point"][g-1])
        blob_point_one_run[idx][0] = data["class"]+1

    return confidence_one_run, choice_one_run, countword_one_run, point_target_one_run, blob_point_one_run, pointblob_one_run, point_results_one_run, class_one_run

for run in range(num_runs):
    path = 'model_runs/' + model_name + '_run_' + str(run + 1) 
    for i, iteration in enumerate(iter_list):
        confidence_all_runs[run, i], choice_all_runs[run, i], countword_all_runs[run, i], point_target_all_runs[run, i], blob_point_all_runs[run, i], pointblob_all_runs[run, i],point_results_all_runs[run, i], class_all_runs[run, i] = fill_matrix(path, iteration)

# Count
stop_idx=np.ones([num_runs, num_iters, num_imgs])*(-1)

for i in range(num_runs):
    for j in range(num_iters):
        for k in range (num_imgs):
            for p in range (1, output_size+1):
                if countword_all_runs[i][j][k][p]==0: # record end position
                    stop_idx[i][j][k]=p
                    break # stop at the first zero 

count_results=np.zeros([num_runs, num_iters, 9, 11])  
count_class=np.ones([num_runs, num_iters, num_imgs])*(-1) #-1: not-well-formed answer (string error)

for i in range(num_runs):
    for j in range(num_iters):
        for k in range(num_imgs):
            # not-well-formed
            if stop_idx[i][j][k]==-1: # no "I'm done!" signal
                count_results[i][j][int(countword_all_runs[i][j][k][0])-1][10]+=1
            # all zeros
            elif stop_idx[i][j][k]==1:
                count_results[i][j][int(countword_all_runs[i][j][k][0])-1][0]+=1 
                count_class[i][j][k]=0
            else:
                for p in range(1,int(stop_idx[i][j][k])): 
                    if p!=countword_all_runs[i][j][k][p]:
                        count_results[i][j][int(countword_all_runs[i][j][k][0])-1][10]+=1
                        break
                    elif p==stop_idx[i][j][k]-1:
                        idx=int(max(countword_all_runs[i][j][k][1:int(stop_idx[i][j][k])+1]))
                        count_results[i][j][int(countword_all_runs[i][j][k][0])-1][idx]+=1
                        count_class[i][j][k]=stop_idx[i][j][k]-1

# How high the network can count? (0.67)
high_cot = np.zeros([num_runs, num_iters]) 
for i in range(num_runs):
    for j in range(num_iters):
        k=0
        while (k<output_size-1): 
            if count_results[i][j][k][k+1]/test_trials>=0.67:
                high_cot[i][j]+=1
                k+=1
            else:
                k=output_size

# Count Accuracy
count_accuracy = np.zeros([num_runs, output_size-1, num_iters])
for i in range(num_runs):
    for j in range(output_size-1):
        for k in range(num_iters):
            count_accuracy[i][j][k]=count_results[i][k][j][j+1]/test_trials

# How high the network can count? (0.90)
high_cot_2 = np.zeros([num_runs, num_iters]) 
for i in range(num_runs):
    for j in range(num_iters):
        k=0
        while (k<output_size-1): 
            if count_results[i][j][k][k+1]/test_trials>=0.90:
                high_cot_2[i][j]+=1
                k+=1
            else:
                k=output_size

# Count Accuracy
count_accuracy_2 = np.zeros([num_runs, output_size-1, num_iters])
for i in range(num_runs):
    for j in range(output_size-1):
        for k in range(num_iters):
            count_accuracy_2[i][j][k]=count_results[i][k][j][j+1]/test_trials

# Point
pstop_idx=np.ones([num_runs, num_iters, num_imgs])*(-1)
for i in range(num_runs):
    for j in range(num_iters):
        for k in range (num_imgs):
            for p in range (2, output_size+1):
                if blob_point_all_runs[i][j][k][p]!=-1 and blob_point_all_runs[i][j][k][p]==blob_point_all_runs[i][j][k][p-1]: # record end position
                    pstop_idx[i][j][k]=p
                    break # stop at the first repeated word

point_results=np.zeros([num_runs, num_iters, 9, 10]) 

for i in range(num_runs):
    for j in range(num_iters):
        for k in range(num_imgs):
            # not-well-formed
            if pstop_idx[i][j][k]==-1: # no "I'm done!" signal
                point_results[i][j][int(blob_point_all_runs[i,j,k,0])-1][9]+=1     
            else:
                for p in range(1,int(pstop_idx[i][j][k])): 
                    if p!=blob_point_all_runs[i][j][k][p]:
                        point_results[i][j][int(blob_point_all_runs[i,j,k,0])-1][9]+=1
                        break
                    elif p==pstop_idx[i][j][k]-1:
                        idx=int(max(blob_point_all_runs[i][j][k][1:int(pstop_idx[i][j][k])+1]))
                        point_results[i][j][int(blob_point_all_runs[i,j,k,0])-1][idx-1]+=1

# Point Accuracy
point_accuracy = np.zeros([num_runs, output_size-1, num_iters])
for i in range(num_runs):
    for j in range(output_size-1):
        for k in range(num_iters):
            point_accuracy[i][j][k]=point_results[i][k][j][j]/test_trials

# Coding Types of Errors
error = np.zeros([num_runs, num_iters, 8])
# error[0]: correct; error[1]: string error; error[2]: done error; error[3]: skip; error[4]: stop short; error[5]: double count; error[6]:continue; error[7]: Others
for i in range(num_runs):
    for j in range(num_iters):
        for k in range(num_imgs):
            cot_idx=int(count_class[i][j][k]) # the last count word produced by NN
            if cot_idx==countword_all_runs[i][j][k][0]: # correct
                error[i][j][0]+=1   
            elif cot_idx ==-1: # string error (count words in an incorrect order)
                error[i][j][1]+=1
            elif cot_idx ==0: # done error (say "I'm done!" at the very beginning)
                error[i][j][2]+=1
            elif cot_idx<countword_all_runs[i][j][k][0]: # count too far
                if blob_point_all_runs[i][j][k][cot_idx]==blob_point_all_runs[i][j][k][cot_idx+1]:
                    if blob_point_all_runs[i][j][k][cot_idx]>cot_idx:
                        for p in range(1, cot_idx):
                            if blob_point_all_runs[i][j][k][p]>=blob_point_all_runs[i][j][k][p+1]:
                                error[i][j][7]+=1 # others 
                                break
                            elif p==cot_idx-1:
                                error[i][j][3]+=1 # skip error
                    elif blob_point_all_runs[i][j][k][cot_idx]==cot_idx:
                        for p in range(1, cot_idx):
                            if blob_point_all_runs[i][j][k][p]+1!=blob_point_all_runs[i][j][k][p+1]:
                                error[i][j][7]+=1 # others
                                break
                            elif p==cot_idx-1:
                                error[i][j][4]+=1 # stop short
                else:
                    error[i][j][7]+=1 # others  
            else: # count too short
                if blob_point_all_runs[i][j][k][cot_idx]==blob_point_all_runs[i][j][k][cot_idx+1]:
                    if blob_point_all_runs[i][j][k][cot_idx]<cot_idx:
                        for p in range(1, cot_idx):
                            if blob_point_all_runs[i][j][k][p]>blob_point_all_runs[i][j][k][p+1]:
                                error[i][j][7]+=1 # others
                                break
                            elif blob_point_all_runs[i][j][k][p]<blob_point_all_runs[i][j][k][p+1]-1:
                                error[i][j][7]+=1 # others
                                break
                            elif p==cot_idx-1:
                                error[i][j][5]+=1 # double count  
                else:
                    for q in range(1, int(countword_all_runs[i][j][k][0]+1)):
                        if blob_point_all_runs[i][j][k][q]!=pointblob_all_runs[i][j][k][q]:
                            error[i][j][7]+=1 # others       
                        elif q==countword_all_runs[i][j][k][0]:
                            error[i][j][6]+=1 # continue

np.save(data_directory + "count_results_hist", count_results)
np.save(data_directory + "count_class_hist", count_class)
np.save(data_directory + "high_cot_hist", high_cot)
np.save(data_directory + "high_cot_2_hist", high_cot_2)
np.save(data_directory + "count_accuracy_hist", count_accuracy)
np.save(data_directory + "count_accuracy_2_hist", count_accuracy)
np.save(data_directory + "point_results_hist", point_results)
np.save(data_directory + "point_accuracy_hist", point_accuracy)
np.save(data_directory + "error_hist", error)
