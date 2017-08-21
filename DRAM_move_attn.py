#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
#from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import load_teacher
from model_settings import learning_rate, glimpses, batch_size, min_edge, max_edge, min_blobs, max_blobs, model_name

FLAGS = tf.flags.FLAGS

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if not os.path.exists("model_runs"):
    os.makedirs("model_runs")

folder_name = "model_runs/" + model_name

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

start_restore_index = 0 

sys.argv = [sys.argv[0], "true", "true", "true", "true", "true", "true",
folder_name + "/classify_log.csv",
folder_name + "/classifymodel_" + str(start_restore_index) + ".ckpt",
folder_name + "/classifymodel_",
folder_name + "/zzzdraw_data_5000.npy",
"false", "true", "false", "false", "true"]
print(sys.argv)

pretrain_iters = 10000000
train_iters = 20000000000
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
dims = [40, 200]#[10, 10]
img_size = dims[1]*dims[0] # canvas size
read_n = 15 # read glimpse grid width/height
read_size = read_n*read_n
z_size = max_blobs - min_blobs + 1 # QSampler output size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])


## BUILD MODEL ## 

REUSE = None

input_tensor = tf.placeholder(tf.float32, shape=(batch_size, glimpses, img_size))
count_tensor = tf.placeholder(tf.float32, shape=(batch_size, glimpses, z_size + 1))
target_tensor = tf.placeholder(tf.float32, shape=(batch_size, glimpses, 2))

#batch_size is 77 when running DRAM
#input_tensor = tf.placeholder(tf.float32, shape=(77, glimpses, img_size))
#count_tensor = tf.placeholder(tf.float32, shape=(77, glimpses, z_size))
#target_tensor = tf.placeholder(tf.float32, shape=(77, glimpses, 2))

lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer())
    b=tf.get_variable("b", [output_dim], initializer=tf.random_normal_initializer())
    return tf.matmul(x,w)+b


def filterbank(gx, gy, sigma2, delta, N):
    min_dim = min(dims[0],dims[1])    
    mu_x = np.zeros([N,N])
    for i in range((N+1)//2):
        mu_x[i,i:N-i] = np.linspace(-sum(delta[i:(N-1)//2]), sum(delta[i:(N-1)//2]), N-2*i)
        mu_x[i+1:(N+1)//2,i] = mu_x[i,i]
        mu_x[i+1:(N+1)//2,N-1-i] = mu_x[i,N-1-i]
    
    mu_x[(N-1)//2,(N-1)//2]=0

    for i in range((N+1)//2,N):
        mu_x[i,:] = mu_x[N-1-i,:]

    mu_y = np.zeros([N,N])
    for i in range((N+1)//2):
        mu_y[i,i:N-i] = -sum(delta[i:(N-1)//2])
        mu_y[i:(N+1)//2,i] = np.linspace(-sum(delta[i:(N-1)//2]), 0, (N+1)//2 - i)
        mu_y[i:(N+1)//2,N-1-i] = np.linspace(-sum(delta[i:(N-1)//2]), 0, (N+1)//2 - i)
                                                    
    mu_y[(N-1)//2,(N-1)//2]=0

    for i in range((N+1)//2,N):
        mu_y[i,:] = -mu_y[N-1-i,:]
   
    # mu_x = np.reshape([mu_x]*batch_size, (batch_size, N, N))
    # gx = tf.reshape(gx, [batch_size, 1, 1])
    mu_x = gx + mu_x # batch_size x N x N
    
    # mu_y = np.reshape([mu_y]*batch_size, (batch_size, N, N))
    # gy = tf.reshape(gy, [batch_size, 1, 1])
    mu_y = gy + mu_y # batch_size x N x N
    
    a = tf.reshape([tf.cast(tf.range(dims[0]), tf.float32)]*N, [N, 1, -1])
    b = tf.reshape([tf.cast(tf.range(dims[1]), tf.float32)]*N, [N, 1, -1])

    mu_x = tf.reshape(mu_x, [N, N, 1])
    mu_y = tf.reshape(mu_y, [N, N, 1])
    sigma2 = tf.cast(tf.reshape(sigma2, [-1, N, 1]), tf.float32)
    
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2)) # N x N x dims[0]
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # N x N x dims[1]
    #Fx = tf.reshape(Fx, [batch_size, N, dims[0]]) # batch_size x N x A
    #Fy = tf.reshape(Fy, [batch_size, N, dims[1]]) # batch_size x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx, Fy, mu_x, mu_y


def attn_window(scope,h_dec,N, predx=None, predy=None, DO_SHARE=False):
    if DO_SHARE:
        with tf.variable_scope(scope,reuse=True):
            params=linear(h_dec, 3)
    else:
        with tf.variable_scope(scope,reuse=REUSE):
            params=linear(h_dec, 3)

    gx_,gy_,log_gamma=tf.split(params, 3, 1) # batch_size x 1
    gx = (dims[0]+1)/2*(gx_+1)
    gy = (dims[1]+1)/2*(gy_+1)
   
    if predx is not None and predy is not None:
        gx=tf.reshape(predx, [batch_size, 1])
        gy=tf.reshape(predy, [batch_size, 1])

    # constrain gx and gy
    max_gx = np.array([dims[0]]) 
    tmax_gx = tf.convert_to_tensor(max_gx, dtype=tf.float32)
    gx = tf.minimum(gx, tmax_gx)

    min_gx = np.array([0]) 
    tmin_gx = tf.convert_to_tensor(min_gx, dtype=tf.float32)
    gx = tf.maximum(gx, tmin_gx)

    max_gy = np.array([dims[1]]) 
    tmax_gy = tf.convert_to_tensor(max_gy, dtype=tf.float32)
    gy = tf.minimum(gy, tmax_gy)

    min_gy = np.array([0]) 
    tmin_gy = tf.convert_to_tensor(min_gy, dtype=tf.float32)
    gy = tf.maximum(gy, tmin_gy) 
    
    #gx_list[glimpse] = gx
    #gy_list[glimpse] = gy

    pdelta=np.logspace(1, (N-1)//2 - 2, (N-1)//2 - 2, base=1.3)
    pdelta=np.append(1,(np.append(1,pdelta)))
    delta=pow(3,pdelta)
    delta=np.append(np.append(delta[::-1],delta[0]), delta) # sum(delta[0:7])=109.89
    sigma2=delta*delta/4 # sigma=delta/2
    
    #delta_list[glimpse] = delta
    #sigma_list[glimpse] = sigma2

    Fx, Fy, mu_x, mu_y = filterbank(gx, gy, sigma2, delta, N)
    gamma = tf.exp(log_gamma)
    delta = tf.reshape(tf.convert_to_tensor(delta), [1,-1])
    sigma2 = tf.reshape(tf.convert_to_tensor(sigma2), [1,-1])
    return Fx, Fy, mu_x, mu_y, gamma, gx, gy, delta


## READ ## 
#def read_no_attn(x,x_hat,h_dec_prev):
#    return x, stats


#def read(x, h_dec_prev, position=None):
def read(x, h_dec_prev, pred_x=None, pred_y=None):

    Fx, Fy, mu_x, mu_y, gamma, gx, gy, delta = attn_window("read", h_dec_prev, read_n, pred_x, pred_y)
    stats = Fx, Fy, gamma
    new_stats = mu_x, mu_y, gx, gy, delta

    def filter_img(img, Fx, Fy, gamma, N):
        glimpse = tf.reshape(img[0][0], [1,1,1]) 
        img = tf.reshape(img,[-1, dims[1], dims[0]])
        for i in range(N):
            for j in range(N):
                gg=tf.matmul(tf.reshape(Fy[i][j][:], [1,1,-1]), tf.matmul(img, tf.reshape(Fx[i][j][:], [1,-1,1])))
                glimpse = tf.concat([glimpse,gg], 0)
        glimpse = tf.reshape(glimpse,[-1, N*N+1])[0, 1:N*N+1]
        return glimpse * tf.reshape(gamma, [-1,1])

    xr = filter_img(x, Fx, Fy, gamma, read_n) # batch_size x (read_n*read_n)
    return xr, new_stats # concat along feature axis

#read = read_attn if FLAGS.read_attn else read_no_attn


def encode(input, state):
    """
    run LSTM
    input: cat(read, h_dec_prev)
    state: previous encoder state
    returns: (output, new_state)
    """

    with tf.variable_scope("encoder/LSTMCell", reuse=REUSE):
        return lstm_enc(input, state)


def decode(input, state):
    """
    run LSTM
    input: cat(write, h_dec_prev)
    state: previous decoder state
    returns: (output, new_state)
    """

    with tf.variable_scope("decoder/LSTMCell", reuse=REUSE):
        return lstm_dec(input, state)


def write(h_dec):
    with tf.variable_scope("write", reuse=REUSE):
        return linear(h_dec,img_size)


def convertTranslated(images):
    newimages = []
    for k in range(batch_size):
        image = images[k * dims[0] * dims[1] : (k + 1) * dims[0] * dims[1]]
        newimages.append(image)
    return newimages


def dense_to_one_hot(labels_dense, num_classes=z_size):
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


## STATE VARIABLES ##############
#outputs = [0] * glimpses

# initial states
h_dec_prev = tf.zeros((batch_size,dec_size))
enc_state = lstm_enc.zero_state(batch_size, tf.float32)
dec_state = lstm_dec.zero_state(batch_size, tf.float32)

#classifications = list()
#position_qualities = list()
#count_qualities = list()
viz_data = list()

#gx_list = [0] * glimpses 
#gy_list = [0] * glimpses
#sigma_list = [0] * glimpses
#delta_list = [0] * glimpses

#Using w and b to predict the starting point
#with tf.variable_scope("starting_point", reuse=None):
#    predict_x, predict_y = tf.split(linear(input_tensor[:, 0], 2), 2, 1)

#current_blob = target_tensor[:, 0]
#current_x, current_y = tf.split(current_blob, num_or_size_splits=2, axis=1)
#current_cnt = count_tensor[:, 0]
current_x = tf.constant(10, dtype=tf.float32, shape=[batch_size, 1])
current_y = tf.constant(10, dtype=tf.float32, shape=[batch_size, 1])
current_cnt = tf.zeros(dtype=tf.float32, shape=[batch_size, z_size + 1])
next_index = 1#0
next_blob_position = target_tensor[:, next_index]
next_blob_cnt = count_tensor[:, next_index]
reward_position_list = list()
reward_count_list = list()
accuracy_count_list = list()
train_ops = list()
train_ops2 = list()
predict_x_list = list()
predict_cnt_list = list()
target_cnt_list = list()
target_x_list = list()
testing = False
#testing = True

while next_index < glimpses:

    target_x, target_y = tf.split(next_blob_position, num_or_size_splits=2, axis=1)
    target_cnt = next_blob_cnt  

    if testing:
        r, new_stats = read(input_tensor[:, next_index-1], h_dec_prev) # when testing, target_x and target_y are None
    else:            
        #set current attn window center to current blob center and perform read

        # don't have to use the same weights and matrices as in attn_window below
        r, new_stats = read(input_tensor[:, next_index-1], h_dec_prev, current_x, current_y)
        #c = linear(tf.concat([input_tensor[:, next_index-1], h_dec_prev, current_cnt], 1), 1)

    h_enc, enc_state = encode(tf.concat([r, h_dec_prev], 1), enc_state) 

    with tf.variable_scope("z", reuse=REUSE):
        #z = linear(tf.concat([current_cnt, h_enc], 1), z_size)
        z = linear(h_enc, z_size)
    h_dec, dec_state = decode(z, dec_state)
    _, _, _, _, _, attn_x, attn_y, _ = attn_window("read", h_dec, read_n, DO_SHARE=True)
    predict_x, predict_y = attn_x, attn_y
    with tf.variable_scope("hidden", reuse=REUSE):
        hidden = tf.nn.relu(linear(tf.concat([current_cnt, h_dec], 1), 256))
        #hidden = tf.nn.relu(linear(h_dec, 256))

    with tf.variable_scope("count", reuse=REUSE):
        predict_cnt = tf.nn.softmax(linear(hidden, z_size + 1))

    #pred_cnt_idx = tf.arg_max(predict_cnt, 1) * tf.cast(tf.reduce_sum(target_cnt), dtype=tf.int64)
    pred_cnt_idx = tf.arg_max(predict_cnt, 1)
    targ_cnt_idx = tf.arg_max(target_cnt, 1)
    target_cnt_list.append(targ_cnt_idx)
    predict_cnt_list.append(pred_cnt_idx)
    
    cnt_precision = tf.log(predict_cnt + 1e-5) * target_cnt # softmax cross entropy
    reward_cnt = tf.reduce_sum(cnt_precision, 1)
    cntquality = reward_cnt
    reward_count_list.append(reward_cnt)
    
    accuracy_weight = tf.reduce_sum(target_cnt)
    #accuracy_weight = tf.cast(accuracy_weight, dtype=tf.float32)
    #accuracy_cnt = tf.cast(tf.equal(pred_cnt_idx, targ_cnt_idx), tf.float32)*accuracy_weight
    accuracy_cnt = tf.cast(tf.equal(pred_cnt_idx, targ_cnt_idx), dtype=tf.float32)
    accuracy_count_list.append(accuracy_cnt)
    

    # change reward so that multiple traces are rewarded
    #reward = tf.constant(1, shape=[77,1], dtype=tf.float32)  - tf.nn.relu(((tf.abs(predict_x - target_x) - max_edge/2)**2 + (tf.abs(predict_y - target_y)-max_edge)**2)/(dims[0]*dims[1]))

    # new reward 
    in_blob_x = tf.logical_and(tf.less(target_x - max_edge/2, predict_x), tf.less(predict_x, target_x + max_edge/2))
    in_blob_y = tf.logical_and(tf.less(target_y - max_edge/2, predict_y), tf.less(predict_y, target_y + max_edge/2))
    in_blob = tf.logical_and(in_blob_x, in_blob_y)
    intensity = (predict_x - target_x)**2 + (predict_y - target_y)**2
    reward = tf.where(in_blob, 200 - intensity, -200 - intensity)
    #reward = tf.cond(in_blob, lambda: tf.nn.relu((predict_x - target_x)**2 + (predict_y - target_y)**2 - (max_edge/2)**2),  lambda: -200-tf.nn.relu((predict_x - target_x)**2 + (predict_y - target_y)**2 - (max_edge/2)**2))


    predict_x_list.append(predict_x[0])
    target_x_list.append(target_x[0])
    reward_position_list.append(reward)
 
    posquality = reward
    
    mu_x, mu_y, gx, gy, delta = new_stats
    #print(tf.equal(gx, target_x))
    stats = gx, gy, delta
    viz_data.append({
        "r": r,
        "h_dec": h_dec,
        "predict_x": predict_x,
        "predict_y": predict_y,
        "predict_cnt": predict_cnt,
        "gx": gx,
        "gy": gy,
        "stats": stats,
        "mu_x": tf.squeeze(mu_x, 2)[0], # batch_size x N
        "mu_y": tf.squeeze(mu_y, 2)[0],
    })

    #poscost = -posquality
    #cntcost = -cntquality
    cost = -posquality/10 - cntquality

    ## OPTIMIZER #################################################
    #why can't we use the same optimizer across all the glimpses?
    #with tf.variable_scope("optimizer", reuse=REUSE):
    with tf.variable_scope("pos_optimizer" + str(next_index), reuse=None):
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1)
        grads = optimizer.compute_gradients(cost)

        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        train_op = optimizer.apply_gradients(grads)
        train_ops.append(train_op)
    
    next_index = next_index + 1
    if next_index < glimpses:
        current_x = target_x
        current_y = target_y
        current_cnt = target_cnt
        
        next_blob_position = target_tensor[:,next_index]
        next_blob_cnt = count_tensor[:, next_index]

    REUSE=True

    REUSE=True
    h_dec_prev = h_dec

avg_position_reward = tf.reduce_mean(reward_position_list)
avg_count_reward = tf.reduce_mean(reward_count_list)
avg_count_accuracy = tf.reduce_mean(accuracy_count_list)
#one_reward = tf.reshape(rewards[0][0], [1])
#relu_num = tf.reduce_mean(relus)
predict_x_average = tf.reduce_mean(tf.convert_to_tensor(predict_x_list))
target_x_average = tf.reduce_mean(tf.convert_to_tensor(target_x_list))
#print("predict_x_average", tf.reduce_mean(tf.convert_to_tensor(predict_x_list)))
#avg_reward = tf.constant(1, shape=[1], dtype=tf.float32)

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))


def evaluate():
    testing = True
    data = load_teacher.Teacher()
    data.get_test(1)
    batches_in_epoch = len(data.explode_images) // batch_size
    accuracy_position = 0
    accuracy_count = 0
    
    for i in range(batches_in_epoch):
        xtrain, _, _, explode_counts, ytrain = data.next_explode_batch(batch_size)
        feed_dict = { input_tensor: xtrain, count_tensor: explode_counts, target_tensor: ytrain }
        #feed_dict = { input_tensor: xtrain, target_tensor: ytrain }
        fetch_accuracy = []
        fetch_accuracy.extend([avg_position_reward, avg_count_accuracy])
        r = sess.run(fetch_accuracy, feed_dict=feed_dict)
        r_position, r_count = r
        accuracy_position += r_position
        accuracy_count += r_count
    
    accuracy_position /= batches_in_epoch
    accuracy_count /= batches_in_epoch

    print("POSITION ACCURACY: " + str(accuracy_position))
    print("COUNT ACCURACY: " + str(accuracy_count))
    testing = False
    return accuracy_position, accuracy_count





if __name__ == '__main__':
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)
    
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    
    if restore:
        saver.restore(sess, load_file)

    train_data = load_teacher.Teacher()
    train_data.get_train(1)
    fetches2=[] 
    #fetches2.extend([predict_cnt_list, target_cnt_list, avg_position_reward, avg_count_reward, train_op, train_op2, predict_x_average, target_x_average, train_ops, train_ops2])
    fetches2.extend([predict_cnt_list, target_cnt_list, avg_position_reward, avg_count_reward, train_op, predict_x_average, target_x_average, train_ops])
    #fetches2.extend([avg_reward, train_op, relu_num, predict_x_average, target_x_average, train_ops])

    start_time = time.clock()
    extra_time = 0

    for i in range(start_restore_index, train_iters):
        xtrain, _, _, explode_counts, ytrain = train_data.next_explode_batch(batch_size)
        feed_dict = { input_tensor: xtrain, count_tensor: explode_counts, target_tensor: ytrain }
        #feed_dict = { input_tensor: xtrain, target_tensor: ytrain }
        results = sess.run(fetches2, feed_dict=feed_dict) 
        #reward_fetched, _, relu_fetched, prex, tarx, _ = results
        predict_count_list, target_cnt_list, reward_position_fetched, reward_count_fetched, _, prex, tarx, _ = results
        #predict_count_list, target_cnt_list, reward_position_fetched, reward_count_fetched, _, _, prex, tarx, _, _= results
        
        if i%100 == 0:
            print(predict_count_list)
            #print(target_cnt_list)
            print("iter=%d : Reward: %f" % (i, reward_position_fetched))
            print("iter=%d : Reward: %f" % (i, reward_count_fetched))
            #print("One of the rewards: %f" % a_reward_fetched)
            #print("Average relu: %f" % relu_fetched)
            sys.stdout.flush()
 
            train_data = load_teacher.Teacher()
            train_data.get_train(1)

            if i%1000 == 0:
                start_evaluate = time.clock()
                test_accuracy = evaluate()
                saver = tf.train.Saver(tf.global_variables())
                #sess.run(predict_x_list[0])
                #sess.run(target_x_list[0])
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
                    settings_file.write("min_blobs = " + str(min_blobs) + ", ")
