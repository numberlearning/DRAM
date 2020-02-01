#!/usr/bin/env py thon
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
#from scipy import misc
import time
import sys
import load_input
from model_settings import learning_rate, batch_size, img_height, img_width, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean("read_attn", False, "enable attention for reader")

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
folder_name + "/classify_log.csv",
folder_name + "/classifymodel_" + str(start_restore_index) + ".ckpt",
folder_name + "/classifymodel_",
folder_name + "/zzzdraw_data_5000.npy",
"false", "true", "false", 
"false", # restore
"true"]
print(sys.argv)

train_iters = 3000000#20000000000
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
draw_file = sys.argv[10]
pretrain = str2bool(sys.argv[11]) #False
classify = str2bool(sys.argv[12]) #True
pretrain_restore = False
translated = str2bool(sys.argv[13])
dims = [img_height, img_width]
img_size = dims[1]*dims[0] # canvas size
read_n = 15  # read glimpse grid width/height
read_size = read_n*read_n
output_size = max_blobs_train - min_blobs_train + 1 # QSampler output size
h_size = 50 #250
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])
# delta, sigma2
delta_1=max(dims[0],dims[1])/(read_n-1) 
sigma2_1=delta_1*delta_1/4 # sigma=delta/2 

## BUILD MODEL ## 

REUSE = None

x = tf.placeholder(tf.float32,shape=(batch_size, img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))
mask = tf.placeholder(tf.float32, shape=(h_size))

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

def filterbank(gx, gy, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x_1 = gx + (grid_i - N / 2 + 0.5) * delta_1 # eq 19 batch_size x N
    mu_y_1 = gy + (grid_i - N / 2 + 0.5) * delta_1 # eq 20 batch_size x N
    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[0]
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[1]

    mu_x_1 = tf.reshape(mu_x_1, [-1, N, 1]) # batch_size x N x 1
    mu_y_1 = tf.reshape(mu_y_1, [-1, N, 1])
    Fx_1 = tf.exp(-tf.square(a - mu_x_1) / (2*sigma2_1)) # batch_size x N x dims[0]
    Fy_1 = tf.exp(-tf.square(b - mu_y_1) / (2*sigma2_1)) # batch_size x N x dims[1]
    # normalize, sum over A and B dims
    Fx_1=Fx_1/tf.maximum(tf.reduce_sum(Fx_1,2,keep_dims=True),eps)
    Fy_1=Fy_1/tf.maximum(tf.reduce_sum(Fy_1,2,keep_dims=True),eps)
    return Fx_1,Fy_1

def attn_window(scope,N):
    
    gx=dims[0]/2  
    gy=dims[1]/2 
    gx=np.reshape([gx]*batch_size, [batch_size,1])
    gy=np.reshape([gy]*batch_size, [batch_size,1])
    Fx_1, Fy_1 = filterbank(gx, gy, N)
    return Fx_1, Fy_1, gx, gy


## READ ## 

def read(x):
    Fx_1, Fy_1, gx, gy = attn_window("read", read_n)
    stats = Fx_1, Fy_1
    new_stats = gx, gy
    
    def filter_img(img, Fx_1, Fy_1, N):
        Fxt_1 = tf.transpose(Fx_1, perm=[0,2,1])
        # img: 1 x img_size
        img = tf.reshape(img,[-1, dims[1], dims[0]])
        fimg_1 = tf.matmul(Fy_1, tf.matmul(img, Fxt_1))
        fimg_1 = tf.reshape(fimg_1,[-1, N*N])
        # normalization (if do norm, Pc will be nan)
        # scalar_1 = tf.reshape(tf.reduce_max(fimg_1, 1), [batch_size, 1])
        # fimg_1 = fimg_1/tf.reduce_max(fimg_1, 1, keep_dims=True)
        fimg = fimg_1 
        return fimg

    xr = filter_img(x, Fx_1, Fy_1, read_n) # batch_size x (read_n*read_n)
    return xr, new_stats # concat along feature axis

## STATE VARIABLES ##############
# initial states
r, stats = read(x) 
rr=r
maxr=tf.reduce_max(rr,1, keep_dims=True)
classifications = list()
hiddens1 = list()
hiddens2 = list()
hiddens3 = list()
hiddens4 = list()
hiddens5 = list()

with tf.variable_scope("hidden1",reuse=REUSE):
    hidden1 = tf.nn.relu(linear(r, h_size))
    masked_hidden1 = tf.multiply(hidden1, mask)
    masked_hidden1 = tf.reshape(masked_hidden1, [batch_size, h_size])

    hiddens1.append({
        "hidden1":hidden1,
        "masked_hidden1":masked_hidden1,
        })

with tf.variable_scope("hidden2",reuse=REUSE): 
    hidden2 = tf.nn.relu(linear(hidden1, h_size))
    masked_hidden2 = tf.multiply(hidden2, mask)
    masked_hidden2 = tf.reshape(masked_hidden2, [batch_size, h_size])
 
    hiddens2.append({
        "hidden2":hidden2,
        "masked_hidden2":masked_hidden2,
        })

with tf.variable_scope("hidden3",reuse=REUSE):
    hidden3 = tf.nn.relu(linear(hidden2, h_size))
    masked_hidden3 = tf.multiply(hidden3, mask)
    masked_hidden3 = tf.reshape(masked_hidden3, [batch_size, h_size])

    hiddens3.append({
        "hidden3":hidden3,
        "masked_hidden3":masked_hidden3,
        })

with tf.variable_scope("hidden4",reuse=REUSE):
    hidden4 = tf.nn.relu(linear(hidden3, h_size))
    masked_hidden4 = tf.multiply(hidden4, mask)
    masked_hidden4 = tf.reshape(masked_hidden4, [batch_size, h_size])

    hiddens4.append({
        "hidden4":hidden4,
        "masked_hidden4":masked_hidden4,
        })

with tf.variable_scope("hidden5",reuse=REUSE):
    hidden5 = tf.nn.relu(linear(hidden4, h_size))
    masked_hidden5 = tf.multiply(hidden5, mask)
    masked_hidden5 = tf.reshape(masked_hidden5, [batch_size, h_size])
 
    hiddens5.append({
        "hidden5":hidden5,
        "masked_hidden5":masked_hidden5,
        })

with tf.variable_scope("output",reuse=REUSE):
    classification = tf.nn.softmax(linear(hidden5, output_size))
#    raw_output = linear(hidden, output_size)
#    masked_raw_output = linear(masked_hidden, output_size)
#    masked_classification = tf.nn.softmax(linear(masked_hidden, output_size))
    classifications.append({
        "classification":classification,
#        "masked_classification":masked_classification,
#        "raw_output":raw_output,
#        "masked_raw_output":masked_raw_output,
        "r":r,
        })

REUSE=True

## LOSE FUNCTION ################################
predquality = -tf.reduce_sum(tf.log(classification + 1e-5) * onehot_labels, 1) # cross-entropy
predcost = tf.reduce_mean(predquality)

correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)

# all-knower
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)

def evaluate():
    data = load_input.InputData()
    data.get_test(1, min_blobs_test, max_blobs_test) # MT
    batches_in_epoch = len(data.images) // batch_size
    accuracy = 0
    sumlabels = np.zeros(output_size)
 
    for i in range(batches_in_epoch):
        nextX, nextY, nextZ = data.next_batch(batch_size)
        sumlabels += np.sum(nextY, 0) 
        feed_dict = {x: nextX, onehot_labels: nextY}
        c, mr, iimg, r, pred, cor = sess.run([classification, maxr, rr, reward, prediction, correct], feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch
   
    #print("CLASSIFICATION: " + str(c))
    #print("MAX: " + str(mr))
    #print("img: " + str(iimg)) 
    print("ACCURACY: " + str(accuracy))
    #print("LabelSums: " + str(sumlabels))  
    print("CORRECT: " + str(cor)) 
    print("PREDICTION: " + str(pred))
    return accuracy

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

    train_data = load_input.InputData()
    train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
    fetches2=[]
    fetches2.extend([reward, predcost, train_op])

    start_time = time.clock()
    extra_time = 0
    
    sum_rwd = 0
    sum_pc = 0

    for i in range(start_restore_index, train_iters+1):
        xtrain, ytrain, ztrain = train_data.next_batch(batch_size)
        results = sess.run(fetches2, feed_dict = {x: xtrain, onehot_labels: ytrain})
        reward_fetched, predcost_fetched, _ = results
        sum_rwd += reward_fetched # average over 100 batches
        sum_pc += predcost_fetched 

        if i%100==0:
            print("iter=%d : Reward: %f Pc: %f" % (i, sum_rwd/100, sum_pc/100))
            sum_rwd = 0
            sum_pc = 0
            sys.stdout.flush()
            
            if i%1000==0:
                train_data = load_input.InputData()
                train_data.get_train(None, min_blobs_train, max_blobs_train) # MT
     
                
        #if i in [0, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 19200, 25600, 38400, 51200, 76800, 102400, 153600, 204800, 307200, 409600, 614400, 819200, 1000000, 1228800, 1638400, 2000000, 2457600, 3000000, 3276800, 4000000, 4915200, 5000000, 6000000, 6553600, 7000000]:
        if i in [0, 200, 400, 1600, 6400, 25600, 102400, 204800, 409600, 819200,    1000000, 1228800, 1638400, 2000000, 3000000, 4000000, 5000000, 6000000, 6000100]: 
            start_evaluate = time.clock()
            test_accuracy = evaluate()
            saver = tf.train.Saver(tf.global_variables())
            print("Model saved in file: %s" % saver.save(sess, save_file + str(i) + ".ckpt"))
            extra_time = extra_time + time.clock() - start_evaluate
            print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
            if i == 0:
                log_file = open(log_filename, 'w')
                settings_file = open(settings_filename, "w")
                settings_file.write("learning_rate = " + str(learning_rate) + ", ")
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
            log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_accuracy) + "\n")
            log_file.close()
