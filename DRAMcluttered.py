#!/usr/bin/env python



import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
from load_mnist import load_data
import custom_lstm

FLAGS = tf.flags.FLAGS

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


translated = True #str2bool(sys.argv[13]) #True
if translated:
    dims = [100, 100]
else:
    dims = [28, 28]
img_size = dims[1]*dims[0]
read_n = 12
read_size = read_n*read_n
z_size=10
glimpses=10
batch_size=1
enc_size = 256
dec_size = 256
pretrain_iters=10000000
train_iters=10000000
learning_rate=1e-4
eps=1e-8
pretrain = False #str2bool(sys.argv[11]) #False
classify = True #str2bool(sys.argv[12]) #True
pretrain_restore = False
restore = False #str2bool(sys.argv[14]) #True
rigid_pretrain = False
log_filename = "log.csv" #sys.argv[7] #"translatedplain/classify_weird_from_20000_log.csv"
load_file = "" #sys.argv[8] #"translatedplain/drawmodel20000.ckpt"
save_file = "save_" #sys.argv[9] #"translatedplain/classifymodel_weird_from_20000_"
draw_file = "draw_" #sys.argv[10] #"translatedplain/zzzdraw_data_5000.npy"
start_non_restored_from_random = True #str2bool(sys.argv[15])
dist_size = (9, 9)
ORG_SHP = [28, 28]
OUT_SHP = [100, 100]
NUM_DISTORTIONS_DB = 100000
mnist_data = load_data('mnist.pkl')



### create list with distortions
all_digits = np.concatenate([mnist_data['X_train'],
                             mnist_data['X_valid']], axis=0)
all_digits = all_digits.reshape([-1] + ORG_SHP)
num_digits = all_digits.shape[0]

distortions = []
for i in range(NUM_DISTORTIONS_DB):
    rand_digit = np.random.randint(num_digits)
    rand_x = np.random.randint(ORG_SHP[1]-dist_size[1])
    rand_y = np.random.randint(ORG_SHP[0]-dist_size[0])

    digit = all_digits[rand_digit]
    distortion = digit[rand_y:rand_y + dist_size[0],
                       rand_x:rand_x + dist_size[1]]
    assert distortion.shape == dist_size
                       #plt.imshow(distortion, cmap='gray')
                       #plt.show()
    distortions += [distortion]
# print "Created distortions"


## BUILD MODEL ## 

REUSE=None

x = tf.placeholder(tf.float32,shape=(batch_size,img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))
lstm_enc = custom_lstm.LSTMCell(enc_size, read_size+dec_size, state_is_tuple=True) # encoder Op
lstm_dec = custom_lstm.LSTMCell(dec_size, z_size, state_is_tuple=True) # decoder Op

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
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(dims[0]+1)/2*(gx_+1)
    gy=(dims[1]+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(dims[0],dims[1])-1)/(N-1)*tf.exp(log_delta) # batch x N
    ret = list()
    ret.append(filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),))
    ret.append((gx, gy, delta))
    return ret


def read(x,h_dec_prev):
    foo =attn_window("read",h_dec_prev,read_n)
    Fx,Fy,gamma=foo[0]
    stats = foo[0]
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,dims[1],dims[0]])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    return tf.concat(1,[x]), stats # concat along feature axis



def write(h_dec):
    with tf.variable_scope("write",reuse=REUSE):
        return linear(h_dec,img_size)




def add_distortions(digits, num_distortions):
    canvas = np.zeros_like(digits)
    for i in range(num_distortions):
        rand_distortion = distortions[np.random.randint(NUM_DISTORTIONS_DB)]
        rand_x = np.random.randint(OUT_SHP[1]-dist_size[1])
        rand_y = np.random.randint(OUT_SHP[0]-dist_size[0])
        canvas[rand_y:rand_y+dist_size[0],
               rand_x:rand_x+dist_size[1]] = rand_distortion
    canvas += digits
    return np.clip(canvas, 0, 1)



def create_sample(x, output_shp, num_distortions):
    a, b = x.shape
    x_offset = np.random.choice(range(output_shp[1] - a))
    y_offset = np.random.choice(range(output_shp[1] - b))
    
    angle = np.random.choice(range(int(-b*0.5), int(b*0.5)))
    
    output = np.zeros(output_shp)
    x_start = x_offset
    
    x_end = x_start + b
    y_start = y_offset
    y_end = y_start + a
    if y_end > (output_shp[1]-1):
        m = output_shp[1] - y_end
        y_end += m
        y_start += m
    if y_start < 0:
        m = y_start
        y_end -= m
        y_start -= m
    output[y_start:y_end, x_start:x_end] = x
    if num_distortions > 0:
        output = add_distortions(output, num_distortions)
    output = np.reshape(output, [10000])
    return output







def convertTranslated(images):
    newimages = []
    for k in xrange(batch_size):
        image = images[k, :]
        image = np.reshape(image, [28, 28])
        newimages.append(create_sample(image, [100, 100], num_distortions = 8))
    return newimages

def dense_to_one_hot(labels_dense, num_classes=10):
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



outputs=[0] * glimpses
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)


pqs = list()
classifications = list()

for glimpse in range(glimpses):
    r, stats=read(x,h_dec_prev)
    with tf.variable_scope("encoder", reuse=REUSE):
        h_enc, enc_state, enc_gates = lstm_enc(tf.concat(1,[r,h_dec_prev]), enc_state)
    
    with tf.variable_scope("z",reuse=REUSE):
        z=linear(h_enc,z_size)

    with tf.variable_scope("decoder", reuse=REUSE):
        h_dec, dec_state, dec_gates = lstm_dec(z, dec_state)

    with tf.variable_scope("write", reuse=REUSE):
        outputs[glimpse] = x + 0.0000001 * linear(h_dec, img_size)
    h_dec_prev=h_dec

    with tf.variable_scope("hidden1",reuse=REUSE):
        hidden = tf.nn.relu(linear(h_dec_prev, 256))
    with tf.variable_scope("hidden2",reuse=REUSE):
        classification = tf.nn.softmax(linear(hidden, 10))
        classifications.append({
            "classification": classification,
            "stats": stats,
            "r": r,
            "enc_state": enc_state,
            "dec_state": dec_state,
            "enc_gates": enc_gates,
            "dec_gates": dec_gates,
            # "Fx": Fx,
            # "Fy": Fy,
            # "gamma": gamma,    
        })

    pq = tf.log(classification + 1e-5) * onehot_labels
    pq = tf.reduce_mean(pq, 0)
    pqs.append(pq)



    REUSE=True

# print("THIS MANY PQS", len(pqs))
predquality = tf.reduce_mean(pqs)


correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)


def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))



def evaluate():
    data = mnist.input_data.read_data_sets("mnist", one_hot=True).test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0
    
    for i in xrange(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        if translated:
            nextX = convertTranslated(nextX)
        feed_dict = {x: nextX, onehot_labels:nextY}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    # print("ACCURACY: " + str(accuracy))
    return accuracy


x_recons=tf.nn.sigmoid(outputs[-1])

reconstruction_loss=tf.reduce_sum(binary_crossentropy(x,x_recons),1)
reconstruction_loss=tf.reduce_mean(reconstruction_loss)


predcost = -predquality


##################


# optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
# grads=optimizer.compute_gradients(reconstruction_loss)
# for i,(g,v) in enumerate(grads):
#     if g is not None:
#         grads[i]=(tf.clip_by_norm(g,5),v)
# train_op=optimizer.apply_gradients(grads)

varsToTrain = []

if True:#str2bool(sys.argv[1]):
    
    with tf.variable_scope("read",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)



if True:#str2bool(sys.argv[2]):
    
    with tf.variable_scope("encoder/LSTMCell",reuse=True):
        w = tf.get_variable("W_0")
        varsToTrain.append(w)
        b = tf.get_variable("B")
        varsToTrain.append(b)
            



if True:#str2bool(sys.argv[3]):
    
    with tf.variable_scope("z",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)



if True:#str2bool(sys.argv[4]):
    
    with tf.variable_scope("decoder/LSTMCell",reuse=True):
        w = tf.get_variable("W_0")
        varsToTrain.append(w)
        b = tf.get_variable("B")
        varsToTrain.append(b)



if True:#str2bool(sys.argv[5]):
    
    with tf.variable_scope("hidden1",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)


if True:#str2bool(sys.argv[6]):
    
    with tf.variable_scope("hidden2",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)












optimizer2=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
for v in varsToTrain:
    pass
    # print(v.name)
grads2a=optimizer2.compute_gradients(predcost, var_list = varsToTrain)
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



if __name__ == '__main__':
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)
	#sess=tf.InteractiveSession()
    
    saver = tf.train.Saver()
    tf.initialize_all_variables().run()
    if restore:
        saver.restore(sess, load_file)


    if start_non_restored_from_random:
        tf.initialize_variables(varsToTrain).run()




    if not os.path.exists("mnist"):
        os.makedirs("mnist")
    train_data = mnist.input_data.read_data_sets("mnist", one_hot=True).train
    fetches2=[]
    fetches2.extend([reward,train_op2])


    start_time = time.clock()
    extra_time = 0

    for i in range(train_iters):
        xtrain, ytrain =train_data.next_batch(batch_size)
        if translated:
            xtrain = convertTranslated(xtrain)
        feed_dict={x:xtrain, onehot_labels:ytrain}
        results=sess.run(fetches2,feed_dict)
        reward_fetched,_=results
        if i%1000==0:
            print("iter=%d : Reward: %f" % (i, reward_fetched))
            if i %1000==0:
                start_evaluate = time.clock()
                test_accuracy = evaluate()
                saver = tf.train.Saver(tf.all_variables())
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









