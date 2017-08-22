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

eps = 1e-8 # epsilon for numerical stability
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
draw_file = sys.argv[10]
pretrain = str2bool(sys.argv[11]) #False
classify = str2bool(sys.argv[12]) #True
pretrain_restore = False
translated = str2bool(sys.argv[13])
dims = [100,100]
img_size = dims[1]*dims[0] # canvas size
read_n = 15 # read glimpse grid width/height read_n>5 odd number
read_size = read_n*read_n
z_size = max_blobs - min_blobs + 1 # QSampler output size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])


## BUILD MODEL ## 

REUSE = None

def filterbank(gx, gy, sigma2, delta, N):
    min_dim = min(dims[0],dims[1])    
    mu = zeros([N,N])
    for i in range((N+1)//2):
        mu[i,i:N-i] = linspace(-sum(delta[i:(N-1)//2]), sum(delta[i:(N-1)//2]), N-2*i)
        mu[i+1:(N+1)//2,i] = mu[i,i]
        mu[i+1:(N+1)//2,N-1-i] = mu[i,N-1-i]
    
    mu[(N-1)//2,(N-1)//2]=0

    for i in range((N+1)//2,N):
        mu[i,:] = mu[N-1-i,:]
   
    mu_x = gx + mu
    mu_y = gy + mu
    
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
    return Fx,Fy


def attn_window(scope,h_dec,N):
    gx = (dims[0]+1)/2*(gx_+1)
    gy = (dims[1]+1)/2*(gy_+1)
   
    
    gx_list[glimpse] = gx
    gy_list[glimpse] = gy

    pdelta=np.logspace(1, (N-1)//2 - 2, (N-1)//2 - 2, base=1.3)
    pdelta=np.append(1,(np.append(1,pdelta)))
    delta=pow(3,pdelta)
    delta=np.append(np.append(delta[::-1],delta[0]), delta) # sum(delta[0:7])=109.89
    sigma2=delta*delta/4 # sigma=delta/2
    
    delta_list[glimpse] = delta
    sigma_list[glimpse] = sigma2

    ret = list()
    Fx, Fy = filterbank(gx,gy,sigma2,delta,N)
    delta = tf.reshape(tf.convert_to_tensor(delta), [1,-1])
    sigma2 = tf.reshape(tf.convert_to_tensor(sigma2), [1,-1])
    ret.append((Fx,)+(Fy,)+(tf.exp(log_gamma),)+(gx,)+(gy,)+(delta,))
    return ret

## READ ##

def read(x, h_dec_prev, glimpse):
    att_ret = attn_window("read", h_dec_prev, read_n, glimpse)
    stats = Fx, Fy, gamma, gx, gy, delta = att_ret[0]

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
    return xr, stats # concat along feature axis

#read = read_attn if FLAGS.read_attn else read_no_attn


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
outputs = [0] * glimpses

# initial states

classifications = list()
pqs = list()

gx_list = [0] * glimpses 
gy_list = [0] * glimpses
sigma_list = [0] * glimpses
delta_list = [0] * glimpses


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
