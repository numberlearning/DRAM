#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import load_count
from model_settings import learning_rate, batch_size, glimpses, img_height, img_width, p_size, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT
from activation_factor import actfac_1, actfac_2

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
"true", "false", "false", "true",
folder_name + "/prepoint_"] #sys.argv[10]~[14]
print(sys.argv)

train_iters = 600000#20000000000
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
pre_file = sys.argv[14]
classify = str2bool(sys.argv[10]) #True
translated = str2bool(sys.argv[11]) #False
dims = [img_height, img_width]
img_size = dims[1]*dims[0] # canvas size
read_n = 13  # N x N attention window
read_size = read_n*read_n
output_size = max_blobs_train - min_blobs_train + 1
h_point_size = 256
#h_count_size = 256
restore = str2bool(sys.argv[12]) #False
start_non_restored_from_random = str2bool(sys.argv[13]) #True
# delta, sigma2
delta_1=10#max(dims[0],dims[1])*1.5/(read_n-1) 
sigma2_1=delta_1*delta_1/4 # sigma=delta/2 
delta_2=3#max(dims[0],dims[1])/2/(read_n-1)
sigma2_2=delta_2*delta_2/4 # sigma=delta/2
# normfac
normfac_1 = 1.0/np.sqrt(2*np.pi*sigma2_1)
normfac_2 = 1.0/np.sqrt(2*np.pi*sigma2_2)
# mu: filter location
grid_i = tf.reshape(tf.cast(tf.range(read_n), tf.float32), [1, -1])
mu_1 = (grid_i - read_n / 2 + 0.5) * delta_1 # 1 x read_n 
mu_2 = (grid_i - read_n / 2 + 0.5) * delta_2
# (a,b): a point in the input image
a = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[1] width
b = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[0] height

## BUILD MODEL ## 

REUSE = None

x = tf.placeholder(tf.float32,shape=(batch_size, img_size)) # input img
testing = tf.placeholder(tf.bool) # testing state
task = tf.placeholder(tf.bool, shape=(3)) # task state
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))
blob_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses, 2))
size_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
res_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
mask_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
mask_list_T = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
num_list = tf.placeholder(tf.float32, shape=(batch_size))
count_word = tf.placeholder(tf.float32, shape=(batch_size, glimpses, output_size + 1)) # add "I'm done!" signal
lstm_point = tf.contrib.rnn.LSTMCell(h_point_size, state_is_tuple=True) # point OP 
#lstm_count = tf.contrib.rnn.LSTMCell(h_count_size, state_is_tuple=True) # count OP 

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
    #i_idx = tf.maximum(i_idx, tmin_idx)
    #iv_idx = tf.maximum(iv_idx, tmin_idx)
 
    i = tf.ones(i_idx)
    ii = tf.ones(ii_idx)*255 # pointer blob
    iii = tf.ones(iii_idx)
    iv = tf.ones(iv_idx)
    
    pointer = tf.concat([tf.concat([tf.concat([tf.concat([i,ii], 0), iii], 0), ii], 0), iv], 0)
    pointer = tf.reshape(pointer, [1,img_width*img_height])
    x_pointer = x * pointer
   
    def cond(x_pointer):
        maxval = tf.ones(1, tf.float32)*255
        return tf.less(x_pointer[0, tf.argmax(x_pointer, 1)[0]], maxval)[0]

    def body(x_pointer):
        idx = tf.cast(tf.argmax(x_pointer, 1)[0], tf.int32)
        xx = tf.concat([x_pointer[0,0:idx], tf.ones(1)*255], 0)
        xxx = tf.concat([xx, x_pointer[0,idx+1:img_width*img_height]], 0)
        x_pointer = tf.reshape(xxx, [batch_size, img_width*img_height])
        return x_pointer

    x_pointer = tf.while_loop(cond, body, [x_pointer])

    return x_pointer

def filterbank(gx, gy, N):
    mu_x_1 = tf.reshape(gx + mu_1, [-1, N, 1]) # batch_size x N x 1
    mu_y_1 = tf.reshape(gy + mu_1, [-1, N, 1]) 
    mu_x_2 = tf.reshape(gx + mu_2, [-1, N, 1])
    mu_y_2 = tf.reshape(gy + mu_2, [-1, N, 1])
    Fx_1 = normfac_1 * tf.exp(-tf.square(a - mu_x_1) / (2*sigma2_1)) # batch_size x N x dims[0]
    Fy_1 = normfac_1 * tf.exp(-tf.square(b - mu_y_1) / (2*sigma2_1)) # batch_size x N x dims[1]
    Fx_2 = normfac_2 * tf.exp(-tf.square(a - mu_x_2) / (2*sigma2_2)) # batch_size x N x dims[0]
    Fy_2 = normfac_2 * tf.exp(-tf.square(b - mu_y_2) / (2*sigma2_2)) # batch_size x N x dims[1]
    return Fx_1,Fy_1,Fx_2,Fy_2

def attn_window(scope, blob_list, h_point, N, glimpse, gx_prev, gy_prev, testing): 
    with tf.variable_scope(scope,reuse=REUSE):
        params=linear(h_point,3) # batch_size x 3
    res_,gx_,gy_=tf.split(params, 3, 1) # batch_size x 1
 
    res = tf.sigmoid(res_)

    if glimpse == -1:
        gx_ = tf.zeros([batch_size,1])
        gy_ = tf.zeros([batch_size,1])
        glimpse = 0

    # relative distance
    gx_real = gx_prev + gx_
    gy_real = gy_prev + gy_ 

    # constrain gx and gy
    max_gx = np.array([dims[1]-1])
    tmax_gx = tf.convert_to_tensor(max_gx, dtype=tf.float32)
    gx_real = tf.minimum(gx_real, tmax_gx)

    min_gx = np.array([0])
    tmin_gx = tf.convert_to_tensor(min_gx, dtype=tf.float32)
    gx_real = tf.maximum(gx_real, tmin_gx)

    max_gy = np.array([dims[0]-1])
    tmax_gy = tf.convert_to_tensor(max_gy, dtype=tf.float32)
    gy_real = tf.minimum(gy_real, tmax_gy)
    
    min_gy = np.array([0])
    tmin_gy = tf.convert_to_tensor(min_gy, dtype=tf.float32)
    gy_real = tf.maximum(gy_real, tmin_gy) 

    gx = tf.cond(testing, lambda:gx_real, lambda:tf.ones((batch_size, 1))*blob_list[0][glimpse][0])
    gy = tf.cond(testing, lambda:gy_real, lambda:tf.ones((batch_size, 1))*blob_list[0][glimpse][1])
 
    gx_prev = gx
    gy_prev = gy 
    
    Fx_1, Fy_1, Fx_2, Fy_2 = filterbank(gx, gy, N)
    return Fx_1, Fy_1, Fx_2, Fy_2, gx, gy, gx_real, gy_real, res

## READ ## 
def read(x, h_point_prev, glimpse, testing):
    Fx_1, Fy_1, Fx_2, Fy_2, gx, gy, gx_real, gy_real, res = attn_window("read", blob_list, h_point_prev, read_n, glimpse, gx_prev, gy_prev, testing)
    stats = Fx_1, Fy_1, Fx_2, Fy_2
    new_stats = gx, gy, gx_real, gy_real, res
    # x = add_pointer(x, gx, gy, read_n)
  
    def filter_img(img, Fx_1, Fy_1, Fx_2, Fy_2, N):
        Fxt_1 = tf.transpose(Fx_1, perm=[0,2,1])
        Fxt_2 = tf.transpose(Fx_2, perm=[0,2,1])        
        # img: 1 x img_size
        img = tf.reshape(img,[-1, dims[0], dims[1]])
        fimg_1 = tf.matmul(Fy_1, tf.matmul(img, Fxt_1))
        fimg_1 = tf.reshape(fimg_1,[-1, N*N])
        fimg_2 = tf.matmul(Fy_2, tf.matmul(img, Fxt_2))
        fimg_2 = tf.reshape(fimg_2,[-1, N*N])
        # normalization
        # fimg_1 = fimg_1/tf.reduce_max(fimg_1, 1, keep_dims=True) 
        # fimg_2 = fimg_2/tf.reduce_max(fimg_2, 1, keep_dims=True) 
        fimg_1 = fimg_1/actfac_1
        fimg_2 = fimg_2/actfac_2
        fimg = tf.concat([fimg_1, fimg_2], 1) 
        #fimg = fimg_2 # only use the small scale filters 
        return tf.reshape(fimg, [batch_size, -1])

    xr = filter_img(x, Fx_1, Fy_1, Fx_2, Fy_2, read_n) # batch_size x (read_n*read_n)
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

## COUNTER ##
#def counter(input, state):
#    """
#    run LSTM
#    state: previous lstm_cell state
#    input: cat(read, h_prev)
#    returns: (output, new_state)
#    """
#    with tf.variable_scope("count/LSTMCell", reuse=REUSE):
#        return lstm_count(input, state)
#

## STATE VARIABLES ##############
# initial states
gx_prev = tf.zeros((batch_size, 1))
gy_prev = tf.ones((batch_size, 1))*dims[0]/2
h_point_prev = tf.zeros((batch_size, h_point_size))
#h_count_prev = tf.zeros((batch_size, h_count_size))
point_state = lstm_point.zero_state(batch_size, tf.float32)
#count_state = lstm_count.zero_state(batch_size, tf.float32)
zero_tensor = tf.cast(tf.zeros(1), tf.int64)
classifications = list()
points = list()
corrects = list()
counts = list()
Rs = list()
ress = list() # point response
rerrs = list() # response error
pointxs = list()
pointys = list()
predictxs = list()
predictys = list()
teacherxs = list()
teacherys = list()
xxs = list()
yys = list()
rqs = list() # response quality
pqs = list() # point quality
cqs = list() # count quality
blob_point = list()

# blob point
def cond(blb_pot, glimpse):
    nan = tf.constant(0, tf.float32)
    total_glimpse = tf.constant(glimpses, tf.float32)
    return tf.logical_and(tf.less(blb_pot, nan), tf.less(glimpse, total_glimpse))

def body(blb_pot, glimpse):
    g = tf.cast(glimpse, tf.int32)
    in_range_x = tf.logical_and(tf.greater(predict_gx[0,0], blob_list[0][g][0] - size_list[0][g]/2 - 1), tf.less(predict_gx[0,0], blob_list[0][g][0] + size_list[0][g]/2 + 1))
    in_range_y = tf.logical_and(tf.greater(predict_gy[0,0], blob_list[0][g][1] - size_list[0][g]/2 - 1), tf.less(predict_gy[0,0], blob_list[0][g][1] + size_list[0][g]/2 + 1)) 
    in_range = tf.logical_and(in_range_x, in_range_y)
    blb_pot = tf.cond(in_range, lambda:glimpse+1, lambda:tf.constant(-1, tf.float32))
        
    return blb_pot, glimpse+1

for true_glimpse in range(glimpses+1):
    glimpse = true_glimpse - 1 
    r, stats = read(x, h_point_prev, glimpse, testing)
    point_gx, point_gy, predict_gx, predict_gy, point_res = stats
    task_str = tf.reshape(tf.cast(task, tf.float32), [batch_size, -1])
    h_point, point_state = pointer([tf.concat([r, task_str], 1), point_state)
    h_point_prev = h_point
    #h_count, count_state = counter(tf.concat([h_point, task_str], 1), count_state)
    #h_count_prev = h_count

    with tf.variable_scope("output",reuse=REUSE):
        classification = tf.nn.relu(linear(h_point, output_size + 1)) 
        classifications.append({
            "classification":classification,
            "r":r,
        })

    if true_glimpse != 0:
        target_gx = blob_list[0][glimpse][0]
        target_gy = blob_list[0][glimpse][1]
 
        # count word
        correct = tf.arg_max(count_word[0,glimpse], 0)
        count = tf.arg_max(classification, 1)[0]
        corrects.append(correct)
        counts.append(count)
        R = tf.cast(tf.equal(correct, count), tf.float32)
        Rs.append(R)

        # pointer
        ress.append(point_res[0,0])
        teacherxs.append(target_gx)
        teacherys.append(target_gy)
        pointxs.append(point_gx[0,0])
        pointys.append(point_gy[0,0])
        predictxs.append(predict_gx[0,0])
        predictys.append(predict_gy[0,0])
        xx = [predict_gx[0,0], target_gx]
        xxs.append(xx)
        yy = [predict_gy[0,0], target_gy]
        yys.append(yy)

        # response quality
        resquality = -res_list[0,glimpse]*tf.log(point_res+1e-5)-(1-res_list[0,glimpse])*tf.log(1-point_res+1e-5) # cross-entropy
        rq = tf.reduce_mean(resquality)
        rqs.append(rq)

        res_error = tf.abs(res_list[0,glimpse]-point_res)
        rerr = tf.reduce_mean(res_error)
        rerrs.append(rerr)

        # point quality
        in_blob_gx = tf.logical_and(tf.less(target_gx - size_list[0][glimpse]/2, predict_gx), tf.less(predict_gx, target_gx + size_list[0][glimpse]/2))
        in_blob_gy = tf.logical_and(tf.less(target_gy - size_list[0][glimpse]/2, predict_gy), tf.less(predict_gy, target_gy + size_list[0][glimpse]/2))
        in_blob = tf.logical_and(in_blob_gx, in_blob_gy)
        intensity = tf.sqrt((predict_gx - target_gx)**2 + (predict_gy - target_gy)**2)
        #potquality = tf.where(in_blob, tf.reshape(tf.constant(0.0),[-1,1]), intensity) 
        potquality = intensity
        pq = tf.reduce_mean(potquality) 
        pqs.append(pq)
        
        # count quality 
        cntquality = -tf.reduce_sum(tf.log(classification + 1e-5) * count_word[0,glimpse], 1) # cross-entropy
        cq = tf.reduce_mean(cntquality)  
        cqs.append(cq) 

        blb_pot, _ = tf.while_loop(cond, body, (tf.constant(-1, tf.float32), tf.constant(0, tf.float32)))
    
        blob_point.append(blb_pot)
   
        with tf.variable_scope("output",reuse=REUSE):
            points.append({
                "gx":xx,
                "gy":yy,
                "blb_pot":blb_pot,
            })

    REUSE = True
 
## LOSS FUNCTION ################################
predcost1 = tf.reduce_sum(cqs*mask_list[0]) # only count
predcost2 = tf.reduce_sum(pqs*mask_list_T[0])+tf.reduce_sum(rqs*mask_list[0]) # only point
predcost3 = tf.reduce_sum(cqs*mask_list[0]) + tf.reduce_sum(pqs*mask_list_T[0]) + tf.reduce_sum(rqs*mask_list[0]) # count and point 

predcost = tf.cond(task[0], lambda: predcost1, lambda: predcost2)
predcost = tf.cond(task[2], lambda: predcost3, lambda: predcost) 

# all-knower
res_accuracy = tf.reduce_sum(rqs*mask_list_T[0]) / (num_list[0]+1) 
point_accuracy = tf.reduce_sum(pqs*mask_list_T[0]) / (num_list[0]) 
count_accuracy = tf.reduce_sum(Rs*mask_list[0]) / (num_list[0]+1)

def evaluate():
    data = load_count.InputData()
    data.get_test(1, min_blobs_test, max_blobs_test) # MT
    batches_in_epoch = len(data.images) // batch_size
    test_point = 0
    sumlabels = np.zeros(output_size)
 
    for i in range(batches_in_epoch):
        nextX, nextY, nextZ, nextS, nextR, nextM, nextMT, nextN, nextC = data.next_batch(batch_size)
        sumlabels += np.sum(nextY,0) 
        feed_dict = {task: [False, True, False], testing: True, x: nextX, onehot_labels: nextY, blob_list: nextZ, size_list: nextS, res_list: nextR, mask_list: nextM, mask_list_T: nextMT, num_list: nextN, count_word: nextC}
        blbs, ctqs, ptqs, prqs, prerrs, cs, cnt_acr, pot_acr, cnt, cor, potx, poty, prdx, prdy, tchx, tchy, xs, ys = sess.run([blob_point, cqs, pqs, rqs, rerrs, count_word, count_accuracy, point_accuracy, counts, corrects, pointxs, pointys, predictxs, predictys, teacherxs, teacherys, xxs, yys], feed_dict=feed_dict)
        point_maxerror = np.max(ptqs*nextMT[0])
        res_maxerror = np.max(prerrs*nextM[0]) 
        #res_maxerror = np.max(np.exp(prqs*nextM[0])-1)
        if point_maxerror<=2.5 and res_maxerror<=0.05:
            test_point += 1

    print("TESTING!") 
    #print("COUNTWORDS: " + str(cs))
    #print("LabelSums: " + str(sumlabels))  
    #print("POINT_QUALITY: " + str(ptqs)) 
    #print("CORRECT: " + str(cor)) 
    #print("BLOB_POINT: " + str(blbs))
    # print("POINT_X: " + str(potx))
    # print("POINT_Y: " + str(poty))
    # print("PREDICT_X: " + str(prdx))
    # print("TEACHER_X: " + str(tchx))
    # print("PREDICT_Y: " + str(prdy))
    # print("TEACHER_Y: " + str(tchy)) 
    print("PRED_X,TCH_X: " + str(xs))
    print("PRED_Y,TCH_Y: " + str(ys))
    print("point_maxerror: " + str(point_maxerror))
    print("res_maxerror: " + str(res_maxerror))
    print("it_idx: " + str(test_point/batches_in_epoch)) 
    return test_point/batches_in_epoch 

## OPTIMIZER #################################################
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1)
grads = optimizer.compute_gradients(predcost)

for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)
train_op = optimizer.apply_gradients(grads)
    
if __name__ == '__main__':
    
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)
    
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    
    if restore:
        saver.restore(sess, load_file)

    train_data = load_count.InputData()
    train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
    blank_data = load_count.InputData()
    blank_data.get_blank(None, min_blobs_train, max_blobs_train) # MT
    fetches2=[]
    fetches2.extend([ress, blob_point, pointxs, pointys, predictxs, predictys, counts, corrects, count_accuracy, point_accuracy, res_accuracy, predcost, predcost1, train_op])

    start_time = time.clock()
    extra_time = 0
    
    #sum_cnt_accuracy = 0
    #sum_pot_accuracy = 0
    #sum_pc = 0
    sum_cnt_accuracy_1 = 0
    sum_cnt_accuracy_2 = 0
    sum_cnt_accuracy_3 = 0
    
    sum_pot_accuracy_1 = 0
    sum_pot_accuracy_2 = 0
    sum_pot_accuracy_3 = 0

    sum_pc_1 = 0
    sum_pc_2 = 0
    sum_pc_3 = 0

    pot_quality = 0
    res_quality = 0
    pot_Pc = 0
    pot_count = 0
    total_pot_count = 0
    ii = 0
    i_test = 0

    for i in range(start_restore_index*3, train_iters+3):
        xtrain, ytrain, ztrain, strain, rtrain, mtrain, mttrain, ntrain, ctrain = train_data.next_batch(batch_size)
        bxtrain, bytrain, bztrain, bstrain, brtrain, bmtrain, bmttrain, bntrain, bctrain = blank_data.next_batch(batch_size)

        if i<100 or i_test<3:
            total_pot_count += 1
            pot_count+=1
            if i%2==0:
                results = sess.run(fetches2, feed_dict = {task: [False, True, False], testing: False, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain, count_word: ctrain})
            else:
                results = sess.run(fetches2, feed_dict = {task: [False, True, False], testing: True, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain, count_word: ctrain})
            
            ress_fetched, blbs_fetched, potxs_fetched, potys_fetched, prdxs_fetched, prdys_fetched, counts_fetched, corrects_fetched, count_accuracy_fetched, point_accuracy_fetched, res_accuracy_fetched, predcost_fetched, predcost1_fetched, _ = results
            pot_quality += point_accuracy_fetched 
            res_quality += res_accuracy_fetched
            pot_Pc += predcost_fetched

            if i%100==0:
                pot_quality/=100
                res_quality/=100
                pot_Pc/=100

                pot_count=0
                print("iter=%d" % (i))
                print("PrePoint: pot_accuracy: %f, res_accuracy: %f, Pc: %f" % (pot_quality, res_quality, pot_Pc))
                #print("Res_list: " + str(rtrain)) 
                print("CORRECT: " + str(corrects_fetched))
                print("POT_RES: " + str(ress_fetched))
            if i%1000==0:
                train_data = load_count.InputData()
                train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
            if i%100==0:# in [0, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 19200, 25600, 38400, 51200, 76800, 102400, 153600, 204800, 307200, 409600, 614400, 819200, 1000000, 1228800, 1638400, 2000000, 2457600, 3000000, 3276800, 4000000, 4915200, 5000000, 6000000, 6553600, 7000000]:
                start_evaluate = time.clock()
                test_idx = evaluate()
               
                if test_idx >= 0.95:
                    i_test += 1
                else:
                    i_test = 0

                saver = tf.train.Saver(tf.global_variables())
                print("Model saved in file: %s" % saver.save(sess, pre_file + str(i) + ".ckpt"))
                extra_time = extra_time + time.clock() - start_evaluate
                print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
        else:
            if ii%3==0 and ii%2==0:
                results = sess.run(fetches2, feed_dict = {task: [True, False, False], testing: False, x: bxtrain, onehot_labels: bytrain, blob_list: bztrain, size_list: bstrain, res_list: brtrain, mask_list: bmtrain, mask_list_T: bmttrain, num_list: bntrain, count_word: bctrain})
            elif ii%3==0 and ii%2!=0:
                results = sess.run(fetches2, feed_dict = {task: [True, False, False], testing: True, x: bxtrain, onehot_labels: bytrain, blob_list: bztrain, size_list: bstrain, res_list: brtrain, mask_list: bmtrain, mask_list_T: bmttrain, num_list: bntrain, count_word: bctrain})
            elif ii%3==1 and ii%2==0:
                results = sess.run(fetches2, feed_dict = {task: [False, True, False], testing: False, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain, count_word: ctrain})
            elif ii%3==1 and ii%2!=0:
                results = sess.run(fetches2, feed_dict = {task: [False, True, False], testing: True, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain, count_word: ctrain})
            elif ii%3==2 and ii%2==0:
                results = sess.run(fetches2, feed_dict = {task: [False, True, True], testing: False, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain, count_word: ctrain})
            else:
                results = sess.run(fetches2, feed_dict = {task: [False, True, True], testing: True, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain, count_word: ctrain})
        
            ress_fetched, blbs_fetched, potxs_fetched, potys_fetched, prdxs_fetched, prdys_fetched, counts_fetched, corrects_fetched, count_accuracy_fetched, point_accuracy_fetched, res_accuracy_fetched, predcost_fetched, predcost1_fetched, _ = results

            # average over 100 batches
            if ii%3==0:
                sum_cnt_accuracy_1 += count_accuracy_fetched 
                sum_pot_accuracy_1 += point_accuracy_fetched
                sum_pc_1 += predcost_fetched

            elif ii%3==1:
                sum_cnt_accuracy_2 += count_accuracy_fetched 
                sum_pot_accuracy_2 += point_accuracy_fetched
                sum_pc_2 += predcost_fetched 
            else:
                sum_cnt_accuracy_3 += count_accuracy_fetched 
                sum_pot_accuracy_3 += point_accuracy_fetched
                sum_pc_3 += predcost_fetched 

            if ii%300==0 or (ii-1)%300==0 or (ii-2)%300==0:
                #if i==0 or i==1 or i==3:
                #    if i==0:
                #        print("iter=%d" % (i))
                #        print("TASK1: cnt_accuracy: %f, Pc: %f" % (sum_cnt_accuracy_1, sum_pc_1))
                        #print("PRED_X, TCH_X: " + str(xs_fetched)) 
                #    elif i==1: 
                #        print("TASK2: pot_accuracy: %f, Pc: %f" % (sum_pot_accuracy_2, sum_pc_2))
                #        #print("PRED_X, TCH_X: " + str(xs_fetched)) 
                #    else: 
                #        print("TASK3: cnt_accuracy: %f, pot_accuracy: %f, Pc: %f" % (sum_cnt_accuracy_3, sum_pot_accuracy_3, sum_pc_3))
                        #print("PRED_X, TCH_X: " + str(xs_fetched)) 
              
                if ii%300==0:
                    print("iter=%d" % (ii//3))
                    print("TASK1: cnt_accuracy: %f, Pc: %f" % (sum_cnt_accuracy_1/100, sum_pc_1/100))
                    #print("PRED_X, TCH_X: " + str(xs_fetched)) 
                elif (ii-1)%300==0:
                    print("TASK2: pot_accuracy: %f, Pc: %f" % (sum_pot_accuracy_2/100, sum_pc_2/100))
                    #print("PRED_X, TCH_X: " + str(xs_fetched)) 
                else:
                    print("TASK3: cnt_accuracy: %f, pot_accuracy: %f, Pc: %f" % (sum_cnt_accuracy_3/100, sum_pot_accuracy_3/100, sum_pc_3/100))
                    #print("PRED_X, TCH_X: " + str(xs_fetched)) 
            
                if ii%300==0: 
                    sum_cnt_accuracy_1 = 0
                    sum_pot_accuracy_1 = 0
                    sum_pc_1 = 0

                if (ii-1)%300==0: 
                    sum_cnt_accuracy_2 = 0
                    sum_pot_accuracy_2 = 0
                    sum_pc_2 = 0
           
                if (ii-2)%300==0: 
                    sum_cnt_accuracy_3 = 0
                    sum_pot_accuracy_3 = 0
                    sum_pc_3 = 0

                sys.stdout.flush()
            
                if ii%3000==0:
                    train_data = load_count.InputData()
                    train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
                
            if ((ii-2)/3)%100==0:# in [0, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 19200, 25600, 38400, 51200, 76800, 102400, 153600, 204800, 307200, 409600, 614400, 819200, 1000000, 1228800, 1638400, 2000000, 2457600, 3000000, 3276800, 4000000, 4915200, 5000000, 6000000, 6553600, 7000000]:
                #start_evaluate = time.clock()
                #test_count_accuracy = evaluate()
                saver = tf.train.Saver(tf.global_variables())
                print("Model saved in file: %s" % saver.save(sess, save_file + str(ii//3) + ".ckpt"))
                #extra_time = extra_time + time.clock() - start_evaluate
                #print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
                if ii == 0:
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
                #log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_count_accuracy) + "\n")
                log_file.close()
            ii+=1
