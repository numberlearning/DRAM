#!/usr/bin/env python/
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
folder_name + "/onlypoint_"] #sys.argv[10]~[14]
print(sys.argv)

train_iters = 80000#20000000000
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
output_size = max_blobs_train# - min_blobs_train + 1
h_point_size = 169#338#256
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
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))
blob_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses, 2))
size_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
res_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
mask_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
mask_list_T = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
num_list = tf.placeholder(tf.float32, shape=(batch_size))

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    #JLM: small initial weights instead of N(0,1)
    w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_uniform_initializer(minval=-.1, maxval=.1)) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

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
    
    #gx_ = tf.nn.relu(gx_)
    #gy_ = tf.nn.relu(gy_)
    res = tf.sigmoid(res_) # range (0,1)

    if glimpse == -1:
        gx_ = tf.zeros([batch_size,1])
        gy_ = tf.zeros([batch_size,1])
        glimpse = 0

    # relative distance
    gx_real = gx_prev + gx_
    gy_real = gy_prev + gy_ 

    # constrain gx and gy within the canvas
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
 
    Fx_1, Fy_1, Fx_2, Fy_2 = filterbank(gx, gy, N)
    return Fx_1, Fy_1, Fx_2, Fy_2, gx, gy, gx_real, gy_real, res, gx_

## READ ## 
def read(x, h_point, glimpse, gx_prev, gy_prev, testing):
    Fx_1, Fy_1, Fx_2, Fy_2, gx, gy, gx_real, gy_real, res, gx_ = attn_window("read", blob_list, h_point, read_n, glimpse, gx_prev, gy_prev, testing)
    stats = gx, gy, gx_real, gy_real, res, gx_
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
        fimg_1 = fimg_1/actfac_1
        fimg_2 = fimg_2/actfac_2
        #fimg = tf.concat([fimg_1, fimg_2], 1) 
        fimg = fimg_2 # only use the small scale filters 
        return tf.reshape(fimg, [batch_size, -1])

    xr = filter_img(x, Fx_1, Fy_1, Fx_2, Fy_2, read_n) # batch_size x (read_n*read_n)
    return xr, stats 

def pointer(input, h_size):
    with tf.variable_scope("point/HiddenCell", reuse=REUSE):
        return linear(input, h_size)

## STATE VARIABLES ##############
# initial states
gx_prev = tf.zeros((batch_size, 1))
gy_prev = tf.ones((batch_size, 1))*dims[0]/2
h_point = tf.zeros((batch_size, h_point_size))
resvec = tf.zeros((batch_size, 1))
points = list()
resvecs = list()
resraws = list() # point response raw values
reserrs = list() # point response error
pointxs = list()
pointys = list()
predictxs = list()
predictys = list()
teacherxs = list()
teacherys = list()
xxs = list()
yys = list()
rqs = list() # res quality
pqs = list() # point quality
blob_point = list()
pointx_s = list()
pointx_prevs = list()

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

# Training
for true_glimpse in range(glimpses+1):
    glimpse = true_glimpse - 1 
    r, stats = read(x, h_point, glimpse, gx_prev, gy_prev, testing)
    point_gx, point_gy, predict_gx, predict_gy, point_res, point_gx_ = stats
    h_point = r#tf.nn.relu(r) #pointer(r, h_point_size)
    gx_prev = point_gx
    gy_prev = point_gy

    if true_glimpse != 0:
        target_gx = blob_list[0][glimpse][0]
        target_gy = blob_list[0][glimpse][1]
        
        # res: continue or stop
        resraws.append(point_res[0,0])
        resvec_real = (tf.sign(point_res-0.5)+1.0)/2
        resvec_target = tf.reshape(res_list[0,glimpse], [batch_size, 1]) 
        resvec = tf.cond(testing, lambda:resvec_real, lambda:resvec_target) 
        resvecs.append(resvec[0,0])

        # pointer
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
        pointx_s.append(point_gx_[0,0])
        pointx_prevs.append(gx_prev[0,0])

        # res quality
        resquality = -res_list[0,glimpse]*tf.log(point_res+1e-5)-(1-res_list[0,glimpse])*tf.log(1-point_res+1e-5) # cross-entropy
        rq = tf.reduce_mean(resquality)
        rqs.append(rq)

        res_error = tf.abs(res_list[0,glimpse]-point_res)
        reserr = tf.reduce_mean(res_error)
        reserrs.append(reserr)

        # point quality
        in_blob_gx = tf.logical_and(tf.less(target_gx - size_list[0][glimpse]/2, predict_gx), tf.less(predict_gx, target_gx + size_list[0][glimpse]/2))
        in_blob_gy = tf.logical_and(tf.less(target_gy - size_list[0][glimpse]/2, predict_gy), tf.less(predict_gy, target_gy + size_list[0][glimpse]/2))
        in_blob = tf.logical_and(in_blob_gx, in_blob_gy)
        intensity = tf.sqrt((predict_gx - target_gx)**2 + (predict_gy - target_gy)**2)
        #potquality = tf.where(in_blob, tf.reshape(tf.constant(0.0),[-1,1]), intensity) 
        potquality = intensity
        pq = tf.reduce_mean(potquality) 
        pqs.append(pq)
        
        blb_pot, _ = tf.while_loop(cond, body, (tf.constant(-1, tf.float32), tf.constant(0, tf.float32)))
    
        blob_point.append(blb_pot)
   
        with tf.variable_scope("output",reuse=REUSE):
            points.append({
                "gx":xx,
                "gy":yy,
                "blb_pot":blb_pot,
                "gx_":point_gx_[0,0],
            })

    REUSE = True
 
## LOSS FUNCTION ################################
predcost = tf.reduce_sum(pqs*mask_list_T[0])+tf.reduce_sum(rqs*mask_list[0]) # only point

# all-knower
res_accuracy = tf.reduce_sum(rqs*mask_list_T[0]) / (num_list[0]+1) 
point_accuracy = tf.reduce_sum(pqs*mask_list_T[0]) / (num_list[0]) 

def evaluate():
    data = load_count.InputData()
    data.get_test(1, min_blobs_test, max_blobs_test) # MT
    batches_in_epoch = len(data.images) // batch_size
    sumlabels = np.zeros(output_size)
 
    for i in range(batches_in_epoch):
        nextX, nextY, nextZ, nextS, nextR, nextM, nextMT, nextN, nextC = data.next_batch(batch_size)
        sumlabels += np.sum(nextY,0) 
        feed_dict = {testing: True, x: nextX, onehot_labels: nextY, blob_list: nextZ, size_list: nextS, res_list: nextR, mask_list: nextM, mask_list_T: nextMT, num_list: nextN}
        blbs, ptqs, prqs, prerrs, pot_acr, potx, poty, prdx, prdy, tchx, tchy, xs, ys, x_s, x_prevs, resrs, resvs = sess.run([blob_point, pqs, rqs, reserrs, point_accuracy, pointxs, pointys, predictxs, predictys, teacherxs, teacherys, xxs, yys, pointx_s, pointx_prevs, resraws, resvecs], feed_dict=feed_dict)
        point_maxerror = np.max(ptqs*nextMT[0])
        res_maxerror = np.max(prerrs*nextM[0]) 
        #res_maxerror = np.max(np.exp(prqs*nextM[0])-1)

    print("TESTING!") 
    print("PRED_X_: " + str(x_s))
    print("PRED_X,TCH_X: " + str(xs))
    #print("PRED_X_PREV: " + str(x_prevs))
    print("PRED_Y,TCH_Y: " + str(ys))
    print("RES_RAW: " + str(resrs))
    print("RES_VEC:" + str(resvs))
    print("point_maxerror: " + str(point_maxerror))
    print("res_maxerror: " + str(res_maxerror))

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
    fetches2=[]
    fetches2.extend([resraws, blob_point, pointxs, pointys, predictxs, predictys, resvecs, point_accuracy, res_accuracy, predcost, train_op])

    start_time = time.clock()
    extra_time = 0
    
    sum_cnt_accuracy = 0
    sum_pot_accuracy = 0
    sum_pc = 0

    pot_quality = 0
    res_quality = 0
    pot_Pc = 0
    pot_count = 0
    total_pot_count = 0
    ii = 0

    for i in range(start_restore_index, train_iters+1):
        xtrain, ytrain, ztrain, strain, rtrain, mtrain, mttrain, ntrain, ctrain = train_data.next_batch(batch_size)

        total_pot_count += 1
        pot_count+=1
        if i%2==0:
            results = sess.run(fetches2, feed_dict = {testing: False, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain})
        else:
            results = sess.run(fetches2, feed_dict = {testing: True, x: xtrain, onehot_labels: ytrain, blob_list: ztrain, size_list: strain, res_list: rtrain, mask_list: mtrain, mask_list_T: mttrain, num_list: ntrain})
            
        ress_fetched, blbs_fetched, potxs_fetched, potys_fetched, prdxs_fetched, prdys_fetched, resvecs_fetched, point_accuracy_fetched, res_accuracy_fetched, predcost_fetched, _ = results
        pot_quality += point_accuracy_fetched 
        res_quality += res_accuracy_fetched
        pot_Pc += predcost_fetched

        if i%100==0:
            pot_quality/=100
            res_quality/=100
            pot_Pc/=100

            pot_count=0
            print("iter=%d" % (i))
            print("Point: pot_accuracy: %f, res_accuracy: %f, Pc: %f" % (pot_quality, res_quality, pot_Pc))
            #print("Res_list: " + str(rtrain)) 
            print("POT_RES: " + str(ress_fetched))
        
        if i%1000==0:
            train_data = load_count.InputData()
            train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
            evaluate() 
        if i%100==0: 
            start_evaluate = time.clock()
            saver = tf.train.Saver(tf.global_variables())
            print("Model saved in file: %s" % saver.save(sess, pre_file + str(i) + ".ckpt"))
            extra_time = extra_time + time.clock() - start_evaluate
            print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
