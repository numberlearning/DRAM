#!/usr/bin/env python



import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("write_attn",False, "enable attention for writer")
FLAGS = tf.flags.FLAGS


## MODEL PARAMETERS ## 

translated = True
if translated:
    dims = [100, 100]
else:
    dims = [28, 28]
img_size = dims[1]*dims[0] # the canvas size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
read_n = 5 # read glimpse grid width/height
write_n = 5 # write glimpse grid width/height
read_size = read_n*read_n
write_size = write_n*write_n
z_size=10 # QSampler output si
T=10 # number of glimpses
batch_size=100 # training minibatch size
pretrain_iters=100
train_iters=10000000
learning_rate=1e-3 # learning rate for optimizer
eps=1e-8 # epsilon for numerical stability
pretrain = False
classify = True
pretrain_restore = False
restore = True
rigid_pretrain = False
log_filename = sys.argv[1]
load_file = sys.argv[2]
save_file = "translatedplain/second_test_gq_8000_"
draw_file = "translatedplain/zzzdraw_data_5000.npy"
read_hist = []
params_hist = []

## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))
lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, read_size+dec_size) # encoder Op
lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, z_size) # decoder Op

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
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
        params_hist.append(params)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
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
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    return tf.concat(1,[x]) # concat along feature axis



## ENCODE ## 
def encode(state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=DO_SHARE):
        return lstm_enc(input,state)


## DECODER ## 
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        return lstm_dec(input, state)


## WRITER ##
def write(h_dec):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

def convertTranslated(images):
    newimages = []
    for k in xrange(batch_size):
        image = images[k, :]
        image = np.reshape(image, (28, 28))
        randX = random.randint(0, 72)
        randY = random.randint(0, 72)
        image = np.lib.pad(image, ((randX, 72 - randX), (randY, 72 - randY)), 'constant', constant_values = (0))
        image = np.reshape(image, (100*100))
        newimages.append(image)
    return newimages

def dense_to_one_hot(labels_dense, num_classes=10):
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


## STATE VARIABLES ## 

cs=[0]*T # sequence of canvases
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)


## DRAW MODEL ## 

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
    x_hat=x-tf.sigmoid(c_prev) # error image
    r=read(x,h_dec_prev)
    read_hist.append(r)



    h_enc,enc_state=encode(enc_state,tf.concat(1,[r,h_dec_prev]))
    
    with tf.variable_scope("z",reuse=DO_SHARE):
        z=linear(h_enc,z_size)

    h_dec,dec_state=decode(dec_state,z)
    cs[t]=write(h_dec) # store results
    h_dec_prev=h_dec
    DO_SHARE=True # from now on, share variables

with tf.variable_scope("hidden1",reuse=None):
    hidden = tf.nn.relu(linear(h_dec_prev, 256))
with tf.variable_scope("hidden2",reuse=None):
    classification = tf.nn.softmax(linear(hidden, 10))
predquality = tf.log(classification + 1e-5) * onehot_labels
predquality = tf.reduce_mean(predquality, 0)
correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)

## LOSS FUNCTION ##

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))



def evaluate():
    data = mnist.input_data.read_data_sets(data_directory, one_hot=True).test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0
    
    for i in xrange(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        if translated:
            nextX = convertTranslated(nextX)
        feed_dict = {x: nextX, onehot_labels:nextY}
        r = sess.run(reward_test, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)f
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
reconstruction_loss=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
reconstruction_loss=tf.reduce_mean(reconstruction_loss)


predcost = -predquality


##################


optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(reconstruction_loss)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

varsToTrain = []
with tf.variable_scope("hidden1",reuse=True):
    w = tf.get_variable("w")
    varsToTrain.append(w)
    b = tf.get_variable("b")
    varsToTrain.append(b)
with tf.variable_scope("hidden2",reuse=True):
    w = tf.get_variable("w")
    varsToTrain.append(w)
    b = tf.get_variable("b")
    varsToTrain.append(b)

optimizer2=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads2a=optimizer2.compute_gradients(predcost, var_list = varsToTrain)
grads2b=optimizer2.compute_gradients(predcost)

for i,(g,v) in enumerate(grads2a):
    if g is not None:
        grads2a[i]=(tf.clip_by_norm(g,5),v) # clip gradients
for i,(g,v) in enumerate(grads2b):
    if g is not None:
        grads2b[i]=(tf.clip_by_norm(g,5),v) # clip gradients
if rigid_pretrain:
    train_op2=optimizer2.apply_gradients(grads2a)
else:
    train_op2=optimizer2.apply_gradients(grads2b)



load_sess=tf.Session()

saver = tf.train.Saver()
with load_sess.as_default():
    tf.initialize_all_variables().run()
if restore:
    saver.restore(load_sess, load_file)



lstm_test = tf.nn.rnn_cell.LSTMCell(256, 512)
test_state=lstm_dec.zero_state(batch_size, tf.float32)


SHARE = None

for t in range(T):

    no_grad_r = tf.stop_gradient(read_hist[t])
    no_grad_look_params = tf.stop_gradient(params_hist[t])
    with tf.variable_scope("loc", reuse=SHARE):
        look_hidden=tf.tanh(linear(no_grad_look_params, 256))
    with tf.variable_scope("process_read", reuse=SHARE):
        read_hidden=tf.tanh(linear(no_grad_r, 256))
    with tf.variable_scope("lstm_test", reuse=SHARE):
        lstm_out, test_state = lstm_test(tf.concat(1, [read_hidden, look_hidden]), test_state)
    SHARE = True



with tf.variable_scope("test_hidden1",reuse=None):
    hidden_test = tf.nn.relu(linear(lstm_out, 256))
with tf.variable_scope("test_hidden2",reuse=None):
    classification_test = tf.nn.softmax(linear(hidden_test, 10))
predquality_test = tf.log(classification_test + 1e-5) * onehot_labels
predquality_test = tf.reduce_mean(predquality_test, 0)
correct_test = tf.arg_max(onehot_labels, 1)
prediction_test = tf.arg_max(classification_test, 1)
R_test = tf.cast(tf.equal(correct_test, prediction_test), tf.float32)
reward_test = tf.reduce_mean(R_test)
predcost_test = -predquality_test

optimizer_test=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads_test=optimizer.compute_gradients(predcost_test)
for i,(g,v) in enumerate(grads_test):
    if g is not None:
        grads_test[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op_test=optimizer_test.apply_gradients(grads_test)


sess=tf.InteractiveSession()
with sess.as_default():
    tf.initialize_all_variables().run()

with tf.variable_scope("read",reuse=True):
    assign_op = tf.get_variable("w").assign(load_sess.run(tf.get_variable("w")))
    sess.run(assign_op)
    assign_op = tf.get_variable("b").assign(load_sess.run(tf.get_variable("b")))
    sess.run(assign_op)

with tf.variable_scope("z",reuse=True):
    assign_op = tf.get_variable("w").assign(load_sess.run(tf.get_variable("w")))
    sess.run(assign_op)
    assign_op = tf.get_variable("b").assign(load_sess.run(tf.get_variable("b")))
    sess.run(assign_op)


with tf.variable_scope("encoder/LSTMCell",reuse=True):
    assign_op = tf.get_variable("W_0").assign(load_sess.run(tf.get_variable("W_0")))
    sess.run(assign_op)
    assign_op = tf.get_variable("B").assign(load_sess.run(tf.get_variable("B")))
    sess.run(assign_op)


with tf.variable_scope("decoder/LSTMCell",reuse=True):
    assign_op = tf.get_variable("W_0").assign(load_sess.run(tf.get_variable("W_0")))
    sess.run(assign_op)
    assign_op = tf.get_variable("B").assign(load_sess.run(tf.get_variable("B")))
    sess.run(assign_op)






data_directory = os.path.join(FLAGS.data_dir, "mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data

fetches2=[]
fetches2.extend([reward_test,train_op_test])


start_time = time.clock()
extra_time = 0

for i in range(train_iters):
    read_hist = []
    params_hist = []
    xtrain, ytrain =train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
    if translated:
        xtrain = convertTranslated(xtrain)
    feed_dict={x:xtrain, onehot_labels:ytrain}
    results=sess.run(fetches2,feed_dict)
    reward_fetched,_=results
    if i%100==0:
        print("iter=%d : Reward: %f" % (i, reward_fetched))
        if i %1000==0:
            print(log_filename)
            start_evaluate = time.clock()
            test_accuracy = evaluate()
            saver = tf.train.Saver(tf.all_variables()) # saves variables learned during training
            #ckpt_file=os.path.join(FLAGS.data_dir, save_file + str(i) + ".ckpt")
            #print("Model saved in file: %s" % saver.save(sess,ckpt_file))
            extra_time = extra_time + time.clock() - start_evaluate
            print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
            if i == 0:
                log_file = open(log_filename, 'w')
            else:
                log_file = open(log_filename, 'a')
            log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_accuracy) + "\n")
            log_file.close()







sess.close()

print('Done drawing! Have a nice day! :)')