#!/usr/bin/env python


import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import load_input


FLAGS = tf.flags.FLAGS


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


folder_name = "number_learning_lr_tenth_train_over_n"
start_restore_index = 0

sys.argv = [sys.argv[0], "true", "false", "true", "false", "true", "true",
folder_name + "/classify_log.csv",
folder_name + "/classifymodel_" + str(start_restore_index) + ".ckpt",
folder_name + "/classifymodel_",
folder_name + "/zzzdraw_data_5000.npy",
"false", "true", "false", "false", "true"]
print(sys.argv)

pretrain_iters = 10000000
train_iters = 10000000 # train forever . . .
learning_rate = .2 # learning rate for optimizer
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
load_file = sys.argv[8]
save_file = sys.argv[9]
draw_file = sys.argv[10]
pretrain = str2bool(sys.argv[11]) #False
classify = str2bool(sys.argv[12]) #True
pretrain_restore = False
translated = str2bool(sys.argv[13])
dims = [100, 100]
img_size = dims[1]*dims[0] # canvas size
read_n = 5 # read glimpse grid width/height
read_size = read_n*read_n
z_size = 9 # QSampler output size
glimpses = 10
batch_size = 100 # training minibatch size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])


## BUILD MODEL ## 

REUSE=None

x = tf.placeholder(tf.float32,shape=(batch_size,img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, z_size))
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
    print("filterbank: ")
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    print("grid_i.shape: ", grid_i.shape)

    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20

    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1])
    print("a.shape: ", a.shape)

    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1])
    print("b.shape: ", b.shape)

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


def read(x, h_dec_prev):
    stats = Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n)

    def filter_img(img, Fx, Fy, gamma, N):
        Fxt = tf.transpose(Fx,perm=[0,2,1])
        img = tf.reshape(img,[-1,dims[1],dims[0]])
        glimpse = tf.matmul(Fy, tf.matmul(img,Fxt))
        glimpse = tf.reshape(glimpse,[-1,N*N])
        return glimpse * tf.reshape(gamma,[-1,1])

    x = filter_img(x, Fx, Fy, gamma, read_n) # batch x (read_n*read_n)
    return tf.concat([x], 1), stats # concat along feature axis


def write(h_dec):
    with tf.variable_scope("write",reuse=REUSE):
        return linear(h_dec,img_size)


def convertTranslated(images):
    newimages = []
    for k in range(batch_size):
        image = images[k, :]
        image = np.reshape(image, (28, 28))
        randX = random.randint(0, 72)
        randY = random.randint(0, 72)
        image = np.lib.pad(image, ((randX, 72 - randX), (randY, 72 - randY)), 'constant', constant_values = (0))
        image = np.reshape(image, (100*100))
        newimages.append(image)
    return newimages


def dense_to_one_hot(labels_dense, num_classes=z_size):
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
classifications = []

for glimpse in range(glimpses):
    r, stats = read(x, h_dec_prev)
    with tf.variable_scope("encoder", reuse=REUSE):
        h_enc, enc_state = lstm_enc(tf.concat([r,h_dec_prev], 1), enc_state)
    
    with tf.variable_scope("z",reuse=REUSE):
        z = linear(h_enc,z_size)

    with tf.variable_scope("decoder", reuse=REUSE):
        h_dec, dec_state = lstm_dec(z, dec_state)

    with tf.variable_scope("write", reuse=REUSE):
        outputs[glimpse] = linear(h_dec, img_size)

    h_dec_prev=h_dec
    h_enc_prev=h_enc

    with tf.variable_scope("hidden1",reuse=REUSE):
        hidden = tf.nn.relu(linear(h_dec_prev, 256))
    with tf.variable_scope("hidden2",reuse=REUSE):
        classification = tf.nn.softmax(linear(hidden, z_size))
    pq = tf.log(classification + 1e-5) * onehot_labels
    pq = tf.reduce_mean(pq, 0)
    pqs.append(pq)

    REUSE=True

print("len(pqs): ", len(pqs))
print("The below are a_pq in pqs")
for a_pq in pqs:
    print(a_pq)
print("The above are a_pq in pqs")
predquality = tf.reduce_mean(pqs)
correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)


def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))



def evaluate():
    data = load_input.InputData()
    data.get_test()
    batches_in_epoch = len(data.images) // batch_size
    print("batches_in_epoch: ", batches_in_epoch)
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


if str2bool(sys.argv[1]):
    
    with tf.variable_scope("read",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)


if str2bool(sys.argv[2]):
    
    with tf.variable_scope("encoder/LSTMCell",reuse=True):
        w = tf.get_variable("W_0")
        varsToTrain.append(w)
        b = tf.get_variable("B")
        varsToTrain.append(b)


if str2bool(sys.argv[3]):
    
    with tf.variable_scope("z",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)



if str2bool(sys.argv[4]):
    
    with tf.variable_scope("decoder/LSTMCell",reuse=True):
        w = tf.get_variable("W_0")
        varsToTrain.append(w)
        b = tf.get_variable("B")
        varsToTrain.append(b)



if str2bool(sys.argv[5]):
    
    with tf.variable_scope("hidden1",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)


if str2bool(sys.argv[6]):
    
    with tf.variable_scope("hidden2",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)


###############################################################################


with tf.variable_scope("optimizer2", reuse=None):
    optimizer2=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    for v in varsToTrain:
        print(v.name)
    print(predcost.name)
    grads2a=optimizer2.compute_gradients(predcost, var_list = varsToTrain)
    print(varsToTrain)
    grads2b=optimizer2.compute_gradients(predcost)
    
    for i,(g,v) in enumerate(grads2a):
        if g is not None:
            grads2a[i]=(tf.clip_by_norm(g,5),v)
    for i,(g,v) in enumerate(grads2b):
        if g is not None:
            grads2b[i]=(tf.clip_by_norm(g,5),v)
    if rigid_pretrain:
        train_op2=optimizer2.apply_gradients(grads2a)
    else:
        train_op2=optimizer2.apply_gradients(grads2b)
    

if pretrain:


    if not os.path.exists("mnist"):
        os.makedirs("mnist")
    train_data = load_input.InputData("data")
    # train_data.load_sample()
    train_data.get_train()
    print(train_data.get_length())


    fetches=[]
    fetches.extend([reconstruction_loss,train_op])
    reconstruction_lossses=[0]*pretrain_iters



    sess=tf.InteractiveSession()


    tf.global_variables_initializer().run()
    if pretrain_restore:
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, load_file)



    start_time = time.clock()
    extra_time = 0
    for i in range(pretrain_iters):
        xtrain, ytrain =train_data.next_batch(batch_size)
        if translated:
            xtrain = convertTranslated(xtrain)
        
        feed_dict={x:xtrain, onehot_labels:ytrain}
        results=sess.run(fetches,feed_dict)
        reconstruction_lossses[i],_=results
        if i%100==0:
            print("iter=%d : Reconstr. Loss: %f " % (i,reconstruction_lossses[i]))
            if i %1000==0:
                start_evaluate = time.clock()
                saver = tf.train.Saver(tf.all_variables())
                print("Model saved in file: %s" % saver.save(sess, save_file + str(i) + ".ckpt"))
                extra_time = extra_time + time.clock() - start_evaluate
                print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
    




    canvases=sess.run(outputs,feed_dict)
    canvases=np.array(canvases)

    np.save(draw_file,[canvases,reconstruction_lossses])
    print("Outputs saved in file: %s" % draw_file)



    print("FINISHED PRE-TRAINING")


if classify:
    sess=tf.InteractiveSession()
    
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    if restore:
        saver.restore(sess, load_file)
        # start_restore_index = int(load_file[len(save_file):-len(".ckpt")])


    if start_non_restored_from_random:
        tf.variables_initializer(varsToTrain).run()


    train_data = load_input.InputData("data")
    train_data.get_train()
    fetches2=[]
    fetches2.extend([reward,train_op2])


    start_time = time.clock()
    extra_time = 0


    for i in range(start_restore_index, train_iters):
        xtrain, ytrain =train_data.next_batch(batch_size)
        feed_dict={x:xtrain, onehot_labels:ytrain}
        results=sess.run(fetches2,feed_dict)
        reward_fetched,_=results

        if i%100==0:
            print("iter=%d : Reward: %f" % (i, reward_fetched))

            if i%1000==0:
                # train_data = mnist.input_data.read_data_sets("mnist", one_hot=True).train
                train_data = load_input.InputData("data")
                # train_data.load_sample()
                train_data.get_train()
     
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

