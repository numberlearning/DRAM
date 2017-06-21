#!/usr/bin/env python



import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import matplotlib as mpl
import load_input

mpl.use('TkAgg')
import matplotlib.pyplot as plt


FLAGS = tf.flags.FLAGS

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

sys.argv = [sys.argv[0], "true", "false", "true", "false", "true", "true",
"explore_input/classify_weird_from20000_log.csv",
"explore_input/drawmodel20000.ckpt",
"explore_input/classifymodel_weird_from_20000_",
"explore_input/zzzdraw_data_5000.npy",
"false", "true", "true", "false", "true"]
print(sys.argv[0])
print(sys.argv[1])
translated = False # str2bool(sys.argv[13]) #True Sharon! change to false!!!
if translated:
    dims = [100, 100] # image width, height
else:
    dims = [100, 100] # Sharon! change to 100, 100 from 28, 28
img_size = dims[1]*dims[0] # canvas size
read_n = 5 # read glimpse grid width/height
read_size = read_n*read_n
z_size=9 # QSampler output size #Sharon? change from 10 to 9?
glimpses=10
batch_size= 4 #100 # training minibatch size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
pretrain_iters=10000000
train_iters= 10 #10000000 # train forever . . .
learning_rate=1e-3 # learning rate for optimizer
eps=1e-8 # epsilon for numerical stability
pretrain = str2bool(sys.argv[11]) #False
classify = str2bool(sys.argv[12]) #True
pretrain_restore = False
restore = str2bool(sys.argv[14]) #True
rigid_pretrain = True
log_filename = sys.argv[7] #"translatedplain/classify_weird_from_20000_log.csv"
load_file = sys.argv[8] #"translatedplain/drawmodel20000.ckpt"
save_file = sys.argv[9] #"translatedplain/classifymodel_weird_from_20000_"
draw_file = sys.argv[10] #"translatedplain/zzzdraw_data_5000.npy"
start_non_restored_from_random = str2bool(sys.argv[15])


## BUILD MODEL ## 

REUSE=None

x = tf.placeholder(tf.float32,shape=(batch_size,img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, 9)) # Sharon!, change 10 to 9
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, read_size+dec_size) # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, z_size) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim])
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b



def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=REUSE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params, 5, 1)

    gx=(dims[0]+1)/2*(gx_+1)
    gy=(dims[1]+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(dims[0],dims[1])-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)


def read(x,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,dims[1],dims[0]])
        glimpse=tf.matmul(Fy, tf.matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    return tf.concat([x], 1) # concat along feature axis



def write(h_dec):
    with tf.variable_scope("write",reuse=REUSE):
        return linear(h_dec,img_size)


def dense_to_one_hot(labels_dense, num_classes=9): # Sharon! Change 10 to 9!!!
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



outputs=[0] * glimpses
h_dec_prev=tf.zeros((batch_size,dec_size))
h_enc_prev=tf.zeros((batch_size,enc_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

pqs = []

for glimpse in range(glimpses):
    r=read(x,h_dec_prev)
    with tf.variable_scope("encoder", reuse=REUSE):
        h_enc, enc_state = lstm_enc(tf.concat([r,h_dec_prev], 1), enc_state)
    
    with tf.variable_scope("z",reuse=REUSE):
        z=linear(h_enc,z_size)

    with tf.variable_scope("decoder", reuse=REUSE):
        h_dec, dec_state = lstm_dec(z, dec_state)

    with tf.variable_scope("write", reuse=REUSE):
        outputs[glimpse] = linear(h_dec, img_size)

    h_dec_prev=h_dec
    h_enc_prev=h_enc

    with tf.variable_scope("hidden1",reuse=REUSE):
        hidden = tf.nn.relu(linear(h_dec_prev, 256))
    with tf.variable_scope("hidden2",reuse=REUSE):
        classification = tf.nn.softmax(linear(hidden, 9)) # Sharon! change from 10 to 9
    pq = tf.log(classification + 1e-5) * onehot_labels
    pq = tf.reduce_mean(pq, 0)
    pqs.append(pq)
    
    REUSE=True

predquality = tf.reduce_mean(pqs)
correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)


def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))



def evaluate():
    # data = mnist.input_data.read_data_sets("mnist", one_hot=True).test
    # Sharon! change data to the below two lines
    data = load_input.InputData()
    data.load_sample() # Sharon! psst...should actually be load_test

    batches_in_epoch = len(data.images) // batch_size # Sharon! change _images to images
    print("batches_in_epoch: ", batches_in_epoch)
    print("type(data.images): ", type(data.images))
    print("data.images: ", data.images)
    accuracy = 0
    
    for i in range(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        if translated:
            nextX = convertTranslated(nextX)
        feed_dict = {x: nextX, onehot_labels:nextY}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy


x_recons=tf.nn.sigmoid(outputs[-1])

reconstruction_loss=tf.reduce_sum(binary_crossentropy(x,x_recons),1)
reconstruction_loss=tf.reduce_mean(reconstruction_loss)


predcost = -predquality


##################

with tf.variable_scope("optimizer", reuse=None):
    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(reconstruction_loss)
    for i,(g,v) in enumerate(grads):
        if g is not None:
            grads[i]=(tf.clip_by_norm(g,5),v)
    train_op=optimizer.apply_gradients(grads)

varsToTrain = []


with tf.variable_scope("optimizer2", reuse=None):
    optimizer2=tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    grads2b=optimizer2.compute_gradients(predcost)

    for i,(g,v) in enumerate(grads2b):
        if g is not None:
            grads2b[i]=(tf.clip_by_norm(g,5),v)
    train_op2=optimizer2.apply_gradients(grads2b)


if classify:
    sess=tf.InteractiveSession()
    
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    if restore:
        saver.restore(sess, load_file)

    if start_non_restored_from_random:
        tf.variables_initializer(varsToTrain).run()


    if not os.path.exists("mnist"):
        os.makedirs("mnist")
    # train_data = mnist.input_data.read_data_sets("mnist", one_hot=True).train
    # Sharon! change train_data to the below three lines
    train_data = load_input.InputData()
    train_data.load_sample()
    print(train_data.get_length())



    fetches2=[]
    fetches2.extend([reward,train_op2])


    start_time = time.clock()
    extra_time = 0

    for i in range(train_iters):
        xtrain, ytrain =train_data.next_batch(batch_size)
        if i == 0:
            print("xtrain")
            print(xtrain)
            for img_idx, img in enumerate(xtrain):
                print("image #", img_idx, " ", img)
                print("len(img): ", len(img))
                print("max(img): ", max(img))
                print("image #", img_idx, " ", img)
                load_input.print_img(img)
            print("====================================================")
            print("====================================================")
            print("ytrain")
            print(ytrain)
            for img_idx, label in enumerate(ytrain):
                print("label #", img_idx, " ", label)
        feed_dict={x:xtrain, onehot_labels:ytrain}
        results=sess.run(fetches2,feed_dict)
        reward_fetched,_=results
        if i%100==0:
            print("iter=%d : Reward: %f" % (i, reward_fetched))
            if i %10000==0:
                start_evaluate = time.clock()
                test_accuracy = evaluate()
                saver = tf.train.Saver(tf.global_variables())
                print("Model saved in file: %s" % saver.save(sess, save_file + str(i) + ".ckpt"))
                extra_time = extra_time + time.clock() - start_evaluate
                print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
                if i == 0:
                    log_file = open(log_filename, 'w')
                else:
                    log_file = open(log_filename, 'a')
                log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_accuracy) + "\n")
                log_file.close()


sess.close()

