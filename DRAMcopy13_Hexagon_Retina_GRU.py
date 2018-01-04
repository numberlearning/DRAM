#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
from numpy import *
import os
import random
from scipy import misc
import time
import sys
import load_input
from model_settings import learning_rate, glimpses, batch_size, min_edge, max_edge, min_blobs, max_blobs, model_name

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean("read_attn", False, "enable attention for reader")

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

train_iters = 20000000000
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
draw_file = sys.argv[10]
classify = str2bool(sys.argv[12]) #True
translated = str2bool(sys.argv[13])
dims = [100,100]
img_size = dims[1]*dims[0] # canvas size
read_n = 10  # number of circlesi, read_n > 3
num_filter = 1+6+12+18*(read_n-3) # total number of filters
z_size = max_blobs - min_blobs + 1 # QSampler output size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])


## BUILD MODEL ## 

REUSE = None

x = tf.placeholder(tf.float32,shape=(batch_size, img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, z_size))
gru_enc = tf.contrib.rnn.GRUCell(enc_size) # encoder Op
gru_dec = tf.contrib.rnn.GRUCell(dec_size) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer())
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    #b=tf.get_variable("b", [output_dim], initializer=tf.random_normal_initializer())
    return tf.matmul(x,w)+b


def filterbank(gx, gy, sigma2, delta, N):
    mu = np.zeros([num_filter,2])
    mu[0,0] = 0
    mu[0,1] = 0
    for i in range(1,7):
        mu[i,0] = np.cos(np.pi*((i-1)/3))*sum(delta[1:2])
        mu[i,1] = np.sin(np.pi*((i-1)/3))*sum(delta[1:2])
                    
    for i in range(7,19):
        mu[i,0] = np.cos(np.pi*((i-7)/6))*sum(delta[1:3])
        mu[i,1] = np.sin(np.pi*((i-7)/6))*sum(delta[1:3])
                                    
    for i in range(3, N):
        for j in range(1,18+1):
            mu[18*(i-2)+j,0] = np.cos(np.pi*((j-1)/9))*sum(delta[1:i+1])
            mu[18*(i-2)+j,1] = np.sin(np.pi*((j-1)/9))*sum(delta[1:i+1])
                                                                    
    mu_x = gx + mu[:,0] # batch_size x num_filter
    mu_y = gy + mu[:,1] # batch_size x num_filter

    a = np.reshape([np.arange(dims[0])]*num_filter, (num_filter,-1))
    b = np.reshape([np.arange(dims[1])]*num_filter, (num_filter,-1))
    a = np.reshape([a]*batch_size, (batch_size, num_filter, dims[0]))
    b = np.reshape([b]*batch_size, (batch_size, num_filter, dims[1]))

    mu_x = tf.reshape(mu_x, [batch_size, -1, 1])
    mu_y = tf.reshape(mu_y, [batch_size, -1, 1])
   
    FFx = -tf.square(a - mu_x)
    FFy = -tf.square(b - mu_y)
    Fx = tf.reshape(tf.exp(FFx[:,0,:]/(2*sigma2[0])),[batch_size,1,-1])
    Fy = tf.reshape(tf.exp(FFy[:,0,:]/(2*sigma2[0])),[batch_size,1,-1])

    for i in range(1,7):
        FFFx = tf.reshape(tf.exp(FFx[:,i,:]/(2*sigma2[1])),[batch_size,1,-1])
        FFFy = tf.reshape(tf.exp(FFy[:,i,:]/(2*sigma2[1])),[batch_size,1,-1]) 
        Fx = tf.concat([Fx,FFFx],1)    
        Fy = tf.concat([Fy,FFFy],1)

    for i in range(7,19):
        FFFx = tf.reshape(tf.exp(FFx[:,i,:]/(2*sigma2[2])),[batch_size,1,-1])
        FFFy = tf.reshape(tf.exp(FFy[:,i,:]/(2*sigma2[2])),[batch_size,1,-1]) 
        Fx = tf.concat([Fx,FFFx],1)
        Fy = tf.concat([Fy,FFFy],1)

    for i in range(3,N):
        for j in range(1,18+1):
            FFFx = tf.reshape(tf.exp(FFx[:,18*(i-2)+j,:]/(2*sigma2[i])),[batch_size,1,-1])
            FFFy = tf.reshape(tf.exp(FFy[:,18*(i-2)+j,:]/(2*sigma2[i])),[batch_size,1,-1]) 
            Fx = tf.concat([Fx,FFFx],1)
            Fy = tf.concat([Fy,FFFy],1)

    # normalize
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps) # batch_size x num_filter x dims[0]
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps) # batch_size x num_filter x dims[1]
    return Fx,Fy,mu_x,mu_y

def attn_window(scope,h_dec,N, glimpse):
    with tf.variable_scope(scope,reuse=REUSE):
        params=linear(h_dec,3)
    gx_,gy_,log_gamma=tf.split(params, 3, 1) # batch_size x 1
    gx = (dims[0]+1)/2*(gx_+1)
    gy = (dims[1]+1)/2*(gy_+1)
   
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
    
    gx_list[glimpse] = gx
    gy_list[glimpse] = gy

    pdelta=np.logspace(0, N-4, N-3, base=1.145)
    tdelta=pow(9,pdelta)
    tdelta = np.concatenate(((0,3,6),tdelta),axis=0) # tdelta[9]=141.34, 100*sqrt(2)=141.42
    delta=np.zeros(N)
    for i in range(1,N):
            delta[i] = tdelta[i]-tdelta[i-1]
            delta[0] = delta[1]
   
    sigma2=delta*delta/4 # sigma=delta/2
    
    delta_list[glimpse] = delta
    sigma_list[glimpse] = sigma2

    ret = list()
    Fx, Fy, mu_x, mu_y = filterbank(gx,gy,sigma2,delta,N)
    delta = tf.reshape(tf.convert_to_tensor(delta), [1,-1])
    sigma2 = tf.reshape(tf.convert_to_tensor(sigma2), [1,-1])
    ret.append((Fx,)+(Fy,)+(mu_x,)+(mu_y,)+(tf.exp(log_gamma),)+(gx,)+(gy,)+(delta,))
    return ret

## READ ##
#  def read_no_attn(x,x_hat,h_dec_prev):
#    return x, stats


def read(x, h_dec_prev, glimpse):
    att_ret = attn_window("read", h_dec_prev, read_n, glimpse)
    stats = Fx, Fy, mu_x, mu_y, gamma, gx, gy, delta = att_ret[0]

    def filter_img(img, Fx, Fy, mu_x, mu_y, gamma, N):
        img = tf.reshape(img, [-1, dims[1], dims[0]])
        glimpse = tf.ones([1, num_filter])
        for i in range(batch_size):
            iimg = tf.reshape(img[i], [1, dims[1], dims[0]])
            gglimpse = tf.matmul(tf.reshape(Fy[i,0,:], [1,1,-1]), tf.matmul(iimg, tf.reshape(Fx[i,0,:], [1,-1,1])))
            for j in range(1,num_filter):
                gg = tf.matmul(tf.reshape(Fy[i,j,:], [1,1,-1]), tf.matmul(iimg, tf.reshape(Fx[i,j,:], [1,-1,1])))
                gglimpse = tf.concat([gglimpse,gg], 0)
            gglimpse = tf.reshape(gglimpse,[1, num_filter])
            glimpse = tf.concat([glimpse, gglimpse],0)
        return glimpse[1:batch_size+1] * tf.reshape(gamma, [-1,1])

    xr = filter_img(x, Fx, Fy, mu_x, mu_y, gamma, read_n) # batch_size x (read_n*read_n)
    return xr, stats # concat along feature axis


def encode(input, state):
    """
    run GRU
    state: previous encoder state
    input: cat(read, h_dec_prev)
    returns: (output, new_state)
    """ 
    with tf.variable_scope("encoder/GRUCell", reuse=REUSE):
        return gru_enc(input, state)


def decode(input, state):
    """
    run GRU
    state: previous decoder state
    input: cat(write, h_dec_prev)
    returns: (output, new_state)
    """

    with tf.variable_scope("decoder/GRUCell", reuse=REUSE):
        return gru_dec(input, state)


#def write(h_dec):
#    with tf.variable_scope("write", reuse=REUSE):
#        return linear(h_dec,img_size)
#
#
#def convertTranslated(images):
#    newimages = []
#    for k in range(batch_size):
#        image = images[k * dims[0] * dims[1] : (k + 1) * dims[0] * dims[1]]
#        newimages.append(image)
#    return newimages


def dense_to_one_hot(labels_dense, num_classes=z_size):
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


## STATE VARIABLES ##############
outputs = [0] * glimpses

# initial states
h_dec_prev = tf.zeros((batch_size,dec_size))
enc_state = gru_enc.zero_state(batch_size, tf.float32)
dec_state = gru_dec.zero_state(batch_size, tf.float32)

classifications = list()
pqs = list()

gx_list = [0] * glimpses 
gy_list = [0] * glimpses
sigma_list = [0] * glimpses
delta_list = [0] * glimpses

for glimpse in range(glimpses):
    r, stats = read(x, h_dec_prev, glimpse)
   
    h_enc, enc_state = encode(tf.concat([r, h_dec_prev], 1), enc_state)

    with tf.variable_scope("z",reuse=REUSE):
        z = linear(h_enc, z_size)
    h_dec, dec_state = decode(z, dec_state)
    h_dec_prev = h_dec

    with tf.variable_scope("hidden1",reuse=REUSE):
        hidden = tf.nn.relu(linear(h_dec_prev, 256))
    with tf.variable_scope("output",reuse=REUSE):
        classification = tf.nn.softmax(linear(hidden, z_size))
        classifications.append({
            "classification": classification,
            "stats": stats,
            "r": r,
            "h_dec": h_dec,
        })

    REUSE=True

    pq = tf.log(classification + 1e-5) * onehot_labels
    pq = tf.reduce_mean(pq, 0)
    pqs.append(pq)


predquality = tf.reduce_mean(pqs)
correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)

# all-knower

#R = tf.cast(1 - tf.abs(tf.divide(tf.subtract(correct, prediction), correct)), tf.float32)
R = tf.cast(tf.equal(correct, prediction), tf.float32)

reward = tf.reduce_mean(R)


def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))


def evaluate():
    data = load_input.InputData()
    data.get_test(1)
    batches_in_epoch = len(data.images) // batch_size
    accuracy = 0
    
    for i in range(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        feed_dict = {x: nextX, onehot_labels:nextY}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy


predcost = -predquality


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
    train_data.get_train(1)
    fetches2=[]
    fetches2.extend([reward, train_op])

    start_time = time.clock()
    extra_time = 0

    for i in range(start_restore_index, train_iters):
        xtrain, ytrain = train_data.next_batch(batch_size)
        results = sess.run(fetches2, feed_dict = {x: xtrain, onehot_labels: ytrain})
        reward_fetched, _ = results

        if i%1000==0:
            print("iter=%d : Reward: %f" % (i, reward_fetched))
            if False:# i == 0:
                print("gx_list: ", gx_list)
                print("len(gx_list): ", len(gx_list))
                cont = input("Press ENTER to continue this program. ")

                print("gy_list: ", gy_list)
                print("len(gy_list): ", len(gy_list))
                cont = input("Press ENTER to continue this program. ")
                
                print("sigma_list: ", sigma_list)
                print("len(sigma_list): ", len(sigma_list))
                cont = input("Press ENTER to continue this program. ")

                print("delta_list: ", delta_list)
                print("len(delta_list): ", len(delta_list))
                cont = input("Press ENTER to continue this program. ")

            if False:#i != 0:
                for j in range(glimpses):
                    print("At glimpse " + str(j + 1) + " of " + str(glimpses) + ": ")
                    cont = input("Press ENTER to continue this program. ")
                     
                    for k in range(batch_size):
                        print("At image " + str(k + 1) + " of " + str(batch_size) + " in this batch: ")
                        cont = input("Press ENTER to continue this program. ")
                        gx = gx_list[j][k, :]
                        gy = gy_list[j][k, :]
                        sigma2 = sigma_list[j][k, :]
                        delta = delta_list[j][k, :]
                        print("Here are the parameters (gx, gy, sigma2, delta): ")
                        print(gx.eval(), gy.eval(), sigma2.eval(), delta.eval())
                        cont = input("Press ENTER to continue this program. ")

            sys.stdout.flush()

            if i%1000==0:
                train_data = load_input.InputData()
                train_data.get_train(1)
     
                if i %10000==0:
                    start_evaluate = time.clock()
                    test_accuracy = evaluate()
                    saver = tf.train.Saver(tf.global_variables())
                    #saver.restore(sess, "model_runs/rewrite_filterbank_test001/classifymodel_60000.ckpt") 
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
                    log_file.close()
