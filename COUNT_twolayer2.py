#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import load_count
from model_settings import learning_rate, batch_size, glimpses, img_height, img_width, p_size, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if not os.path.exists("model_runs"):
    os.makedirs("model_runs")

if sys.argv[1] is not None:
        model_name = sys.argv[1]
        
folder_name = "model_runs/" + model_name

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

start_restore_index = 0 

sys.argv = [sys.argv[0], sys.argv[1], "true", "true", "true", "true", "true",
folder_name + "/count_log.csv",
folder_name + "/countmodel_" + str(start_restore_index) + ".ckpt",
folder_name + "/countmodel_",
"true", "false", "false", "true"] #sys.argv[10]~[13]
print(sys.argv)

train_iters = 7000000#20000000000
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
classify = str2bool(sys.argv[10]) #True
translated = str2bool(sys.argv[11]) #False
dims = [img_height, img_width]
img_size = dims[1]*dims[0] # canvas size
read_n = 15  # N x N attention window
read_size = read_n*read_n
output_size = max_blobs_train - min_blobs_train + 1
h_point_size = 256
h_count_size = 256
z_size = 10
restore = str2bool(sys.argv[12]) #False
start_non_restored_from_random = str2bool(sys.argv[13]) #True
# delta, sigma2
delta_1=max(dims[0],dims[1])*2/(read_n-1) 
sigma2_1=delta_1*delta_1/4 # sigma=delta/2 
delta_2=max(dims[0],dims[1])/4/(read_n-1)
sigma2_2=delta_2*delta_2/4 # sigma=delta/2

## BUILD MODEL ## 

REUSE = None

x = tf.placeholder(tf.float32,shape=(batch_size, img_size)) # input img
testing = tf.placeholder(tf.bool) # testing state
e = tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))
blob_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses, 2))
size_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
mask_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
num_list = tf.placeholder(tf.float32, shape=(batch_size))
count_word = tf.placeholder(tf.float32, shape=(batch_size, glimpses, output_size + 1)) # add "I'm done!" signal
lstm_point = tf.contrib.rnn.LSTMCell(h_point_size, state_is_tuple=True) # point OP 
lstm_count = tf.contrib.rnn.LSTMCell(h_count_size, state_is_tuple=True) # count OP 

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    #JLM: small initial weights instead of N(0,1)
    w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_uniform_initializer(minval=-.1, maxval=.1)) 
    #w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer())
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def add_pointer(x, gx, gy, N): 
    gx_idx = tf.reshape(tf.cast(gx, tf.int32), [1])
    gy_idx = tf.reshape(tf.cast(gy, tf.int32), [1])
    
    i_idx = (gy_idx-int(p_size/2))*img_width + (gx_idx-int(p_size/2))
    ii_idx = p_size
    iii_idx = (gx_idx-int(p_size/2)) + img_width - (gx_idx-int(p_size/2)) - p_size
    iv_idx = img_width - (gx_idx-int(p_size/2)) - int(p_size) + img_width*(img_height - ((gy_idx-int(p_size/2)) + int(p_size))) 
   
    # constrain min_idx 
    min_idx = np.array([0])
    tmin_idx = tf.convert_to_tensor(min_idx, dtype=tf.int32)
    i_idx = tf.maximum(i_idx, tmin_idx)
    iv_idx = tf.maximum(iv_idx, tmin_idx)
 
    i = tf.zeros(i_idx)
    ii = tf.ones(ii_idx)*255 # pointer blob
    iii = tf.zeros(iii_idx)
    iv = tf.zeros(iv_idx)
    
    pointer = tf.concat([tf.concat([tf.concat([tf.concat([i,ii], 0), iii], 0), ii], 0), iv], 0)
    pointer = tf.reshape(pointer, [1,-1])
    x_pointer = x * pointer
    return x_pointer

def filterbank(gx, gy, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x_1 = gx + (grid_i - N / 2 + 0.5) * delta_1 # eq 19 batch_size x N
    mu_y_1 = gy + (grid_i - N / 2 + 0.5) * delta_1 # eq 20 batch_size x N
    mu_x_2 = gx + (grid_i - N / 2 + 0.5) * delta_2 
    mu_y_2 = gy + (grid_i - N / 2 + 0.5) * delta_2 
    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[0]
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[1]

    mu_x_1 = tf.reshape(mu_x_1, [-1, N, 1]) # batch_size x N x 1
    mu_y_1 = tf.reshape(mu_y_1, [-1, N, 1])
    mu_x_2 = tf.reshape(mu_x_2, [-1, N, 1]) 
    mu_y_2 = tf.reshape(mu_y_2, [-1, N, 1])
    Fx_1 = tf.exp(-tf.square(a - mu_x_1) / (2*sigma2_1)) # batch_size x N x dims[0]
    Fy_1 = tf.exp(-tf.square(b - mu_y_1) / (2*sigma2_1)) # batch_size x N x dims[1]
    Fx_2 = tf.exp(-tf.square(a - mu_x_2) / (2*sigma2_2)) # batch_size x N x dims[0]
    Fy_2 = tf.exp(-tf.square(b - mu_y_2) / (2*sigma2_2)) # batch_size x N x dims[1]
    # normalize, sum over A and B dims
    Fx_1=Fx_1/tf.maximum(tf.reduce_sum(Fx_1,2,keep_dims=True),eps)
    Fy_1=Fy_1/tf.maximum(tf.reduce_sum(Fy_1,2,keep_dims=True),eps)
    Fx_2=Fx_2/tf.maximum(tf.reduce_sum(Fx_2,2,keep_dims=True),eps)
    Fy_2=Fy_2/tf.maximum(tf.reduce_sum(Fy_2,2,keep_dims=True),eps)
    return Fx_1,Fy_1,Fx_2,Fy_2

def attn_window(scope, blob_list, h_point, N, glimpse, gx_prev, gy_prev, testing): 
    with tf.variable_scope(scope,reuse=REUSE):
        params=linear(h_point,3) # batch_size x 3 
    gx_,gy_,log_gamma=tf.split(params, 3, 1) # batch_size x 1

    # relative distance
    gx_real = gx_prev + gx_
    gy_real = gy_prev + gy_ 

    # constrain gx and gy
    max_gx = np.array([dims[0]-p_size/2])
    tmax_gx = tf.convert_to_tensor(max_gx, dtype=tf.float32)
    gx_real = tf.minimum(gx_real, tmax_gx)

    min_gx = np.array([p_size/2])
    tmin_gx = tf.convert_to_tensor(min_gx, dtype=tf.float32)
    gx_real = tf.maximum(gx_real, tmin_gx)

    max_gy = np.array([dims[1]-p_size/2])
    tmax_gy = tf.convert_to_tensor(max_gy, dtype=tf.float32)
    gy_real = tf.minimum(gy_real, tmax_gy)
    
    min_gy = np.array([p_size/2])
    tmin_gy = tf.convert_to_tensor(min_gy, dtype=tf.float32)
    gy_real = tf.maximum(gy_real, tmin_gy) 

    gx = tf.cond(testing, lambda:gx_real, lambda:tf.ones((batch_size, 1))*blob_list[0][glimpse][0])
    gy = tf.cond(testing, lambda:gy_real, lambda:tf.ones((batch_size, 1))*blob_list[0][glimpse][1])

#    if testing:
#       gx = gx_real
#       gy = gy_real
#    else: # teacher: pull the att_window center to the blob center
#       gx = blob_list[0][glimpse][0]
#       gy = blob_list[0][glimpse][1]
 
    gx_prev = gx
    gy_prev = gy 
    
    gamma = tf.exp(log_gamma) 
    Fx_1, Fy_1, Fx_2, Fy_2 = filterbank(gx, gy, N)
    return Fx_1, Fy_1, Fx_2, Fy_2, gamma, gx, gy, gx_real, gy_real

## READ ## 
def read(x, h_point_prev, glimpse, testing):
    Fx_1, Fy_1, Fx_2, Fy_2, gamma, gx, gy, gx_real, gy_real = attn_window("read", blob_list, h_point_prev, read_n, glimpse, gx_prev, gy_prev, testing)
    stats = Fx_1, Fy_1, Fx_2, Fy_2
    new_stats = gx, gy, gx_real, gy_real
    
    if glimpse != 0:
       x = add_pointer(x, gx, gy, read_n)
  
    def filter_img(img, Fx_1, Fy_1, Fx_2, Fy_2, gamma, N):
        Fxt_1 = tf.transpose(Fx_1, perm=[0,2,1])
        Fxt_2 = tf.transpose(Fx_2, perm=[0,2,1])        
        # img: 1 x img_size
        img = tf.reshape(img,[-1, dims[1], dims[0]])
        fimg_1 = tf.matmul(Fy_1, tf.matmul(img, Fxt_1))
        fimg_1 = tf.reshape(fimg_1,[-1, N*N])
        fimg_2 = tf.matmul(Fy_2, tf.matmul(img, Fxt_2))
        fimg_2 = tf.reshape(fimg_2,[-1, N*N])
        # normalization
        fimg_1 = fimg_1/tf.reduce_max(fimg_1, 1, keep_dims=True) 
        fimg_2 = fimg_2/tf.reduce_max(fimg_2, 1, keep_dims=True) 
        fimg = tf.concat([fimg_1, fimg_2], 1) 
        return fimg * tf.reshape(gamma, [-1,1])

    xr = filter_img(x, Fx_1, Fy_1, Fx_2, Fy_2, gamma, read_n) # batch_size x (read_n*read_n)
    return xr, new_stats # concat along feature axis

## POINTER ##
def pointer(input, state):
    """
    run LSTM
    state: previous lstm_cell state
    input: cat(read, h_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("point/LSTMCell", reuse=REUSE):
        return lstm_point(input, state)

## Q-SAMPLER ##
def sampleQ(h_point):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size) 
    """
    with tf.variable_scope("mu",reuse=REUSE):
        mu=linear(h_point,z_size)
    with tf.variable_scope("sigma",reuse=REUSE):
        logsigma=linear(h_point,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## COUNTER ##
def counter(input, state):
    """
    run LSTM
    state: previous lstm_cell state
    input: cat(read, h_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("count/LSTMCell", reuse=REUSE):
        return lstm_count(input, state)

## STATE VARIABLES ##############
mus,logsigmas,sigmas=[0]*glimpses,[0]*glimpses,[0]*glimpses # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
gx_prev = tf.ones((batch_size, 1))*dims[0]/2
gy_prev = tf.ones((batch_size, 1))*dims[1]/2
h_point_prev = tf.zeros((batch_size, h_point_size))
h_count_prev = tf.zeros((batch_size, h_count_size))
point_state = lstm_point.zero_state(batch_size, tf.float32)
count_state = lstm_count.zero_state(batch_size, tf.float32)
zero_tensor = tf.cast(tf.zeros(1), tf.int64)
classifications = list()
corrects = list()
counts = list()
Rs = list()
pointxs = list()
pointys = list()
predictxs = list()
predictys = list()
teachers = list()
cqs = list() # count quality
pqs = list() # point quality
viz_data = list()

for glimpse in range(glimpses):
    r, stats = read(x, h_point_prev, glimpse, testing)
    point_gx, point_gy, predict_gx, predict_gy = stats
    target_gx = blob_list[0][glimpse][0]
    target_gy = blob_list[0][glimpse][1] 
    h_point, point_state = pointer(tf.concat([r, h_point_prev], 1), point_state)
    z,mus[glimpse],logsigmas[glimpse],sigmas[glimpse]=sampleQ(h_point)
    h_point_prev = h_point
    h_count, count_state = counter(z, count_state)
    h_count_prev = h_count

    with tf.variable_scope("output",reuse=REUSE):
        classification = tf.nn.softmax(linear(h_count, output_size + 1)) # add "I'm done!" tensor
        classifications.append({
            "classification":classification,
            "r":r,
        })
 
    REUSE = True
 
    # count word
    correct = tf.arg_max(count_word[0,glimpse], 0)
    count = tf.arg_max(classification, 1)[0]
    corrects.append(correct)
    counts.append(count) 
    R = tf.cast(tf.equal(correct, count), tf.float32)
    Rs.append(R)

    # pointer
    teacher = [target_gx, target_gy]
    pointxs.append(point_gx[0,0])
    pointys.append(point_gy[0,0])
    predictxs.append(predict_gx[0,0])
    predictys.append(predict_gy[0,0])
    teachers.append(teacher)

    # count reward 
    cntquality = -tf.reduce_sum(tf.log(classification + 1e-5) * count_word[0,glimpse], 1) # cross-entropy
    cq = tf.reduce_mean(cntquality)  
    cqs.append(cq) 

    # point reward
    in_blob_gx = tf.logical_and(tf.less(target_gx - size_list[0][glimpse]/2, predict_gx), tf.less(predict_gx, target_gx + size_list[0][glimpse]/2))
    in_blob_gy = tf.logical_and(tf.less(target_gy - size_list[0][glimpse]/2, predict_gy), tf.less(predict_gy, target_gy + size_list[0][glimpse]/2))
    in_blob = tf.logical_and(in_blob_gx, in_blob_gy)
    intensity = tf.sqrt((predict_gx - target_gx)**2 + (predict_gy - target_gy)**2)
    potquality = tf.where(in_blob, tf.reshape(tf.constant(0.0),[-1,1]), intensity)
    pq = tf.reduce_mean(potquality)
    pqs.append(pq)

## LOSE FUNCTION ################################ 
predcost = (tf.reduce_sum(cqs*mask_list[0]) + tf.reduce_sum(pqs*mask_list[0])) / num_list[0]

# all-knower
#R = tf.cast(tf.equal(correct, prediction), tf.float32)
count_accuracy = tf.reduce_sum(Rs*mask_list[0]) / num_list[0]
point_accuracy = tf.reduce_sum(pqs*mask_list[0]) / num_list[0] 

def evaluate():
    data = load_count.InputData()
    data.get_test(1, min_blobs_test, max_blobs_test) # MT
    batches_in_epoch = len(data.images) // batch_size
    test_count_accuracy = 0
    test_point_accuracy = 0
    sumlabels = np.zeros(output_size)
 
    for i in range(batches_in_epoch):
        nextX, nextY, nextZ, nextS, nextM, nextN, nextC = data.next_batch(batch_size)
        sumlabels += np.sum(nextY,0) 
        feed_dict = {testing: True, x: nextX, onehot_labels: nextY, blob_list: nextZ, size_list: nextS, mask_list: nextM, num_list: nextN, count_word: nextC}
        ctqs, ptqs, cs, cnt_acr, pot_acr, cnt, cor, potx, poty, prdx, prdy, tch = sess.run([cqs, pqs, count_word, count_accuracy, point_accuracy, counts, corrects, pointxs, pointys, predictxs, predictys, teachers], feed_dict=feed_dict)
        test_count_accuracy += cnt_acr
        test_point_accuracy += pot_acr
    
    test_count_accuracy /= batches_in_epoch
    test_point_accuracy /= batches_in_epoch
    print("TESTING!") 
    #print("COUNTWORDS: " + str(cs))
    print("LabelSums: " + str(sumlabels))  
    print("COUNT_QUALITY: " + str(ctqs))
    print("POINT_QUALITY: " + str(ptqs)) 
    # print("CNTACCURACY: " + str(test_count_accuracy)) 
    print("CORRECT: " + str(cor)) 
    print("COUNT: " + str(cnt))
    # print("POINT_X: " + str(potx))
    # print("POINT_Y: " + str(poty))
    print("PREDICT_X: " + str(prdx))
    print("PREDICT_Y: " + str(prdy))
    print("TEACHER: " + str(tch))
   
    return test_count_accuracy, test_point_accuracy 

## OPTIMIZER #################################################
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1)
grads = optimizer.compute_gradients(predcost)

for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)
train_op = optimizer.apply_gradients(grads)
    
if __name__ == '__main__':
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)
    
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    
    if restore:
        saver.restore(sess, load_file)

    train_data = load_count.InputData()
    train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
    fetches2=[]
    fetches2.extend([pointxs, pointys, predictxs, predictys, count_accuracy, point_accuracy, predcost, train_op])

    start_time = time.clock()
    extra_time = 0
    
    sum_cnt_accuracy = 0
    sum_pot_accuracy = 0
    sum_pc = 0
    point_accuracy_prev = 0 
    threshold = 0

    for i in range(start_restore_index, train_iters):
        xtrain, ytrain, ztrain, strain, mtrain, ntrain, ctrain = train_data.next_batch(batch_size)
        if point_accuracy_prev < threshold:
            results = sess.run(fetches2, feed_dict = {testing: False, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, mask_list: mtrain, num_list: ntrain, count_word: ctrain})
        else:
            results = sess.run(fetches2, feed_dict = {testing: True, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, mask_list: mtrain, num_list: ntrain, count_word: ctrain})
        potxs_fetched, potys_fetched, prdxs_fetched, prdys_fetched, count_accuracy_fetched, point_accuracy_fetched, predcost_fetched, _ = results
        sum_cnt_accuracy += count_accuracy_fetched # average over 100 batches
        sum_pot_accuracy += point_accuracy_fetched 
        sum_pc += predcost_fetched 

        if i%100==0:
            print("iter=%d : cnt_accuracy: %f, pot_accuracy: %f, Pc: %f" % (i, sum_cnt_accuracy/100, sum_pot_accuracy/100, sum_pc/100))
            # print("point_x: " + str(potxs_fetched))
            # print("point_y: " + str(potys_fetched))
            # print("predict_x: " + str(prdxs_fetched))
            # print("predict_y: " + str(prdys_fetched))
            point_accuracy_prev = sum_pot_accuracy/100
            sum_cnt_accuracy = 0
            sum_pot_accuracy = 0
            sum_pc = 0
            sys.stdout.flush()
            
            if i%1000==0:
                train_data = load_count.InputData()
                train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
                
        if i in [0, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 19200, 25600, 38400, 51200, 76800, 102400, 153600, 204800, 307200, 409600, 614400, 819200, 1000000, 1228800, 1638400, 2000000, 2457600, 3000000, 3276800, 4000000, 4915200, 5000000, 6000000, 6553600, 7000000]:
            start_evaluate = time.clock()
            test_count_accuracy = evaluate()
            saver = tf.train.Saver(tf.global_variables())
            print("Model saved in file: %s" % saver.save(sess, save_file + str(i) + ".ckpt"))
            extra_time = extra_time + time.clock() - start_evaluate
            print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
            if i == 0:
                log_file = open(log_filename, 'w')
                settings_file = open(settings_filename, "w")
                settings_file.write("learning_rate = " + str(learning_rate) + ", ")
                settings_file.write("glimpses = " + str(glimpses) + ", ") 
                settings_file.write("batch_size = " + str(batch_size) + ", ")
                settings_file.write("min_edge = " + str(min_edge) + ", ")
                settings_file.write("max_edge = " + str(max_edge) + ", ")
                settings_file.write("min_blobs_train = " + str(min_blobs_train) + ", ")
                settings_file.write("max_blobs_train = " + str(max_blobs_train) + ", ")
                settings_file.write("min_blobs_test = " + str(min_blobs_test) + ", ")
                settings_file.write("max_blobs_test = " + str(max_blobs_test) + ", ") 
                settings_file.close()
            else:
                log_file = open(log_filename, 'a')
            log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_count_accuracy) + "\n")
            log_file.close()
