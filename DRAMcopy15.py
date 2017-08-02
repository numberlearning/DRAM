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

# folder_name = "model_runs/baby_blobs"

# folder_name = "model_runs/number_learning_test_graph"
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
dims = [10, 10]
img_size = dims[1]*dims[0] # canvas size
read_n = 10 # read glimpse grid width/height
read_size = read_n*read_n
z_size = max_blobs - min_blobs + 1 # QSampler output size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])


## BUILD MODEL ## 

REUSE = None

input_tensor = tf.placeholder(tf.float32, shape=(batch_size, glimpses, img_size))
#count_tensor = tf.placeholder(tf.float32, shape=(batch_size, glimpses, z_size))
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
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20

    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1])

    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch_size x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy, mu_x, mu_y


def attn_window(scope,h_dec,N, predx=None, predy=None, DO_SHARE=False):
    if DO_SHARE:
        with tf.variable_scope(scope,reuse=True):
            params=linear(h_dec,5)
    else:
        with tf.variable_scope(scope,reuse=REUSE):
            params=linear(h_dec,5)

    gx_, gy_, log_sigma2,log_delta,log_gamma=tf.split(params, 5, 1)

    gx=(dims[0]+1)/2*(gx_+1)
    gy=(dims[1]+1)/2*(gy_+1)

    if predx is not None and predy is not None:
        gx=tf.reshape(predx, [batch_size, 1])
        gy=tf.reshape(predy, [batch_size, 1])

    #  sigma2=tf.exp(log_sigma2)
    delta=(max(dims[0],dims[1])-1)/(N-1)*tf.exp(log_delta) # batch x N

    max_deltas = np.array([5]) # batch_size x 1, where 5 is the max delta
    tmax_deltas = tf.convert_to_tensor(max_deltas, dtype=tf.float32)

    delta = tf.minimum(delta, tmax_deltas)

    sigma2=delta*delta/4 # sigma=delta/2

    #delta_list[glimpse] = delta
    #sigma_list[glimpse] = sigma2

    Fx, Fy, mu_x, mu_y = filterbank(gx, gy, sigma2, delta, N)
    gamma = tf.exp(log_gamma)
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
        Fxt = tf.transpose(Fx, perm=[0,2,1])
        img = tf.reshape(img,[-1, dims[1], dims[0]])
        glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
        glimpse = tf.reshape(glimpse,[-1, N*N])
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
with tf.variable_scope("starting_point", reuse=None):
    predict_x, predict_y = tf.split(linear(input_tensor[:, 0], 2), 2, 1)

current_index = 0
current_blob = target_tensor[:, current_index]
trace_length = tf.size(target_tensor[0])
rewards = list()
train_ops = list()
#relus = list()
predict_x_list = list()
target_x_list = list()

while current_index < glimpses:

    #with tf.variable_scope("target", reuse=REUSE):
    #    target_x = tf.expand_dims(target_tensor[:, 0, 0], 1)

        #target_x = tf.reshape(current_blob[:, 0], [77, 1])
    target_x, target_y = tf.split(current_blob, num_or_size_splits=2, axis=1)
    #target_y = tf.reshape(current_blob[:, 1], [77, 1])

    reward = tf.constant(1, shape=[77,1], dtype=tf.float32)  - tf.nn.relu(((predict_x - target_x)**2 + (predict_y - target_y)**2 - max_edge**2)/100)
    #reward = predict_x - predict_y
    predict_x_list.append(predict_x[0])
    target_x_list.append(target_x[0])
#    relus.append(tf.nn.relu(((predict_x-target_x)**2 + (predict_y-target_y)**2 - max_edge**2)100))
    #print(reward)
    rewards.append(reward)
 
    posquality = reward
    
    #set current attn window center to current blob center and perform read
    r, new_stats = read(input_tensor[:, current_index], h_dec_prev, target_x, target_y)
    #r, new_stats = read(input_tensor[:, current_index], h_dec_prev, predict_x, predict_y)

    h_enc, enc_state = encode(tf.concat([r, h_dec_prev], 1), enc_state) 

    with tf.variable_scope("z", reuse=REUSE):
        z = linear(h_enc, z_size)
    h_dec, dec_state = decode(z, dec_state)
    h_dec_prev = h_dec
    _, _, _, _, _, attn_x, attn_y, _ = attn_window("read", h_dec, read_n, DO_SHARE=True)
    #r1, new_stats1 = read(input_tensor[:, current_index], h_dec_prev)
    #_, _, _, gx, gy, _, = new_stats1
    
    with tf.variable_scope("position", reuse=REUSE):
        #predict_x, predict_y  = tf.split(linear(hidden, 2), 2, 1)
        predict_x, predict_y  = attn_x, attn_y

        mu_x, mu_y, gx, gy, delta = new_stats
        stats = gx, gy, delta
        viz_data.append({
            #"r": r,
            #"h_dec": h_dec,
            "predict_x": predict_x,
            "predict_y": predict_y,
            "stats": stats,
            "mu_x": tf.squeeze(mu_x, 2)[0], # batch_size x N
            "mu_y": tf.squeeze(mu_x, 2)[0],
        })
    
    predcost = -posquality

    ## OPTIMIZER #################################################
    with tf.variable_scope("optimizer" + str(current_index), reuse=None):
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1)
        grads = optimizer.compute_gradients(predcost)

        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        train_op = optimizer.apply_gradients(grads)
        train_ops.append(train_op)
 
    current_index = current_index + 1
    if current_index < glimpses:
        current_blob = target_tensor[:,current_index]

    REUSE=True

avg_reward = tf.reduce_mean(rewards)
#one_reward = tf.reshape(rewards[0][0], [1])
#relu_num = tf.reduce_mean(relus)
predict_x_average = tf.reduce_mean(tf.convert_to_tensor(predict_x_list))
target_x_average = tf.reduce_mean(tf.convert_to_tensor(target_x_list))
#print("predict_x_average", tf.reduce_mean(tf.convert_to_tensor(predict_x_list)))
#avg_reward = tf.constant(1, shape=[1], dtype=tf.float32)

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))


def evaluate():
    data = load_teacher.Teacher()
    data.get_test(1)
    batches_in_epoch = len(data.explode_images) // batch_size
    accuracy = 0
    
    for i in range(batches_in_epoch):
        xtrain, _, _, explode_counts, ytrain = train_data.next_explode_batch(batch_size)
        #feed_dict = { input_tensor: xtrain, count_tensor: explode_counts, target_tensor: ytrain }
        feed_dict = { input_tensor: xtrain, target_tensor: ytrain }
        r = sess.run(avg_reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy





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
    fetches2.extend([avg_reward, train_op, predict_x_average, target_x_average, train_ops])
    #fetches2.extend([avg_reward, train_op, relu_num, predict_x_average, target_x_average, train_ops])

    start_time = time.clock()
    extra_time = 0

    for i in range(start_restore_index, train_iters):
        xtrain, _, _, explode_counts, ytrain = train_data.next_explode_batch(batch_size)
        #feed_dict = { input_tensor: xtrain, count_tensor: explode_counts, target_tensor: ytrain }
        feed_dict = { input_tensor: xtrain, target_tensor: ytrain }
        results = sess.run(fetches2, feed_dict=feed_dict) 
        #reward_fetched, _, relu_fetched, prex, tarx, _ = results
        reward_fetched, _, prex, tarx, _ = results

        if i%100 == 0:
            print("iter=%d : Reward: %f" % (i, reward_fetched))
            #print("One of the rewards: %f" % a_reward_fetched)
            #print("Average relu: %f" % relu_fetched)
            print("Predict x average: %f" % prex)
            print("Target x average: %f" % tarx)
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
                    settings_file.write("max_blobs = " + str(max_blobs) + ", ")
                    settings_file.close()
                else:
                    log_file = open(log_filename, 'a')
                log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_accuracy) + "\n")
