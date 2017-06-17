'''TensorFlow implementation of http://arxiv.org/pdf/1511.06434.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
from numpy import random
from scipy.misc import imsave
from tensorflow.examples.tutorials import mnist
import time

from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
from load_mnist import load_data
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

translated = False
if translated:
    dims = [100, 100]
else:
    dims = [28, 28]
img_size = dims[1]*dims[0]
read_n = 2
read_size = read_n*read_n
z_size=10
glimpses=10
enc_size = 256
dec_size = 256
train_iters = 1000000
batch_size = 100
eps = 1e-8
switch = 10000000
pretrain = True



flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 100, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000, "max epoch")
flags.DEFINE_float("g_learning_rate", 1e-2, "learning rate")
flags.DEFINE_float("d_learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_float("hidden_size", 10, "hidden size")
flags.DEFINE_float("gamma", 1, "gamma")

FLAGS = flags.FLAGS


dist_size = (9, 9)
ORG_SHP = [28, 28]
OUT_SHP = [100, 100]
NUM_DISTORTIONS_DB = 100000
mnist_data = load_data('mnist.pkl.gz')



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
print("Created distortions")

global REUSE
REUSE=None



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
        image = np.reshape(image, (28, 28))
        randX = random.randint(0, 72)
        randY = random.randint(0, 72)
        image = np.lib.pad(image, ((randX, 72 - randX), (randY, 72 - randY)), 'constant', constant_values = (0))
        image = np.reshape(image, (100*100))
        newimages.append(image)
    return newimages


'''
def convertTranslated(images):
    newimages = []
    for k in xrange(batch_size):
        image = images[k, :]
        image = np.reshape(image, [28, 28])
        newimages.append(create_sample(image, [100, 100], num_distortions = 8))
    return newimages
'''

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

def dense_to_one_hot(labels_dense, num_classes=10):
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def encoder(x):
    global REUSE
    
    lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, read_size+dec_size) # encoder Op
    lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, z_size) # decoder Op

    lstm_z = tf.nn.rnn_cell.LSTMCell(z_size, enc_size) # decoder Op
    
    outputs=[0] * glimpses
    h_z_prev=tf.zeros((batch_size,z_size))
    h_dec_prev=tf.zeros((batch_size,dec_size))
    h_enc_prev=tf.zeros((batch_size,enc_size))
    enc_state=lstm_enc.zero_state(batch_size, tf.float32)
    dec_state=lstm_dec.zero_state(batch_size, tf.float32)
    z_state=lstm_z.zero_state(batch_size, tf.float32)
    
    for glimpse in range(glimpses):
        r=read(x,h_dec_prev)
        with tf.variable_scope("encoder", reuse=REUSE):
            h_enc, enc_state = lstm_enc(tf.concat(1,[r,h_dec_prev]), enc_state)
        
        with tf.variable_scope("z", reuse=REUSE):
            z=linear(h_enc,z_size)
        
        with tf.variable_scope("decoder", reuse=REUSE):
            h_dec, dec_state = lstm_dec(z, dec_state)

        h_dec_prev=h_dec
        h_enc_prev=h_enc
        REUSE=True
    return h_dec


def discriminator(input_features):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    return  input_features.fully_connected(1, activation_fn=None).tensor


def discriminator_features(input_tensor):
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten())



def get_discrinator_loss(D1, D2):
    '''Loss for the discriminator network

    Args:
        D1: logits computed with a discriminator networks from real images
        D2: logits computed with a discriminator networks from generated images

    Returns:
        Cross entropy loss, positive samples have implicit labels 1, negative 0s
    '''
    return tf.reduce_mean(tf.nn.relu(D1) - D1 + tf.log(1.0 + tf.exp(-tf.abs(D1)))) + \
        tf.reduce_mean(tf.nn.relu(D2) + tf.log(1.0 + tf.exp(-tf.abs(D2))))



def generator(input_tensor):
    '''Create a network that generates images
    TODO: Add fixed initialization, so we can draw interpolated images

    Returns:
        A deconvolutional (not true deconv, transposed conv2d) network that
        generated images.
    '''

    input_sample = tf.reshape(input_tensor, [FLAGS.batch_size, 1, 1, dec_size])
    
    return (pt.wrap(input_sample).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor
    
    '''
    return (pt.wrap(input_sample).
            deconv2d(13, 128, edges='VALID').
            deconv2d(13, 64, edges='VALID').
            deconv2d(16, 32, stride=2).
            deconv2d(16, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor
            
    '''

def binary_crossentropy(t,o):
    return -(t*tf.log(o+1e-9) + (1.0-t)*tf.log(1.0-o+1e-9))

def get_generator_loss(D2):
    '''Loss for the genetor. Maximize probability of generating images that
    discrimator cannot differentiate.

    Returns:
        see the paper
    '''
    return tf.reduce_mean(tf.nn.relu(D2) - D2 + tf.log(1.0 + tf.exp(-tf.abs(D2))))


def evaluate():
    data = input_data.read_data_sets("mnist", one_hot=True).test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0
    
    for i in xrange(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        if translated:
            nextX = convertTranslated(nextX)
        feed_dict = {input_tensor: nextX, onehot_labels:nextY, learning_rate:0.0}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy
  
if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])
    onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))



    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
            with tf.variable_scope("encoder"):
                encoding = encoder(input_tensor)
            E_params_num = len(tf.trainable_variables())
            with tf.variable_scope("hidden1"):
                hidden = tf.nn.relu(linear(encoding, 256))
            with tf.variable_scope("hidden2"):
                classification = tf.nn.softmax(linear(hidden, 10))
            class_params_num = len(tf.trainable_variables())
            with tf.variable_scope("model"):
                input_features = discriminator_features(input_tensor)  # positive examples
                D1 = discriminator(input_features)
                input_features = input_features.tensor
                D_params_num = len(tf.trainable_variables())
                G = generator(encoding)



            with tf.variable_scope("model", reuse=True):
                gen_features = discriminator_features(G)  # positive examples
                D2 = discriminator(gen_features)
                gen_features = gen_features.tensor
            

    
                    
    reconstruction_loss = binary_crossentropy(tf.sigmoid(input_features), tf.sigmoid(gen_features))
    reconstruction_loss = binary_crossentropy(tf.reshape(tf.sigmoid(G), (batch_size, 28*28)), tf.reshape(input_tensor, (batch_size, 28*28)))
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, 1))
    D_loss = get_discrinator_loss(D1, D2)
    G_loss = FLAGS.gamma * reconstruction_loss + get_generator_loss(D2)
    
    predquality = tf.log(classification + 1e-5) * onehot_labels
    predquality = tf.reduce_mean(predquality, 0)
    correct = tf.arg_max(onehot_labels, 1)
    prediction = tf.arg_max(classification, 1)
    R = tf.cast(tf.equal(correct, prediction), tf.float32)
    reward = tf.reduce_mean(R)
    predcost = -predquality
    
    

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
    params = tf.trainable_variables()
    E_params = params[:E_params_num]
    class_params = params[:class_params_num]
    D_params = params[class_params_num:D_params_num]
    G_params = params[D_params_num:]


    train_encoder = pt.apply_optimizer(optimizer, losses=[reconstruction_loss], regularize=True, include_marked=True, var_list=E_params)
    train_discrimator = pt.apply_optimizer(optimizer, losses=[D_loss], regularize=True, include_marked=True, var_list=D_params)
    train_generator = pt.apply_optimizer(optimizer, losses=[G_loss], regularize=True, include_marked=True, var_list=G_params)

    optimizer2=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer2.compute_gradients(predcost)
    for i,(g,v) in enumerate(grads):
        if g is not None:
            grads[i]=(tf.clip_by_norm(g,5),v)
    train_classifier=optimizer2.apply_gradients(grads)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        #saver.restore(sess, "drawmodel_crazyconv_gan_gamma1_2000.ckpt")
        
        if not os.path.exists("mnist"):
            os.makedirs("mnist")
        '''
        train_data = mnist.input_data.read_data_sets("mnist", one_hot=True).train
    



#train_data = mnist.input_data.read_data_sets("mnist", one_hot=True).train
        fetches2=[]
        fetches2.extend([reward, train_classifier])

    

        for i in range(train_iters):
            xtrain, ytrain = train_data.next_batch(batch_size)
            if translated:
                xtrain = convertTranslated(xtrain)




            feed_dict={input_tensor:xtrain, onehot_labels:ytrain, learning_rate:FLAGS.g_learning_rate}
            results=sess.run(fetches2,feed_dict)
            reward_fetched,_=results
            if i%100==0:
                print("iter=%d : Reward: %f" % (i, reward_fetched))
        
        
        '''
        start_time = time.clock()
        extra_time = 0
        for epoch in range(FLAGS.max_epoch):
            if epoch >= switch:
                pretrain = False

            discriminator_loss = 0.0
            generator_loss = 0.0
            encoder_loss = 0.0
            reward_fetched = 0
            print(str(0))
            if (epoch % 10 == 0):
                log_filename = "crazyconv_gan_gamma1_log_from_2000.csv"
                saver = tf.train.Saver(tf.all_variables())
                start_evaluate = time.clock()
                test_accuracy = evaluate()
                print("Model saved in file: %s" % saver.save(sess, "drawmodel_crazyconv_gan_gamma1_pixel_" + str(100*epoch) + ".ckpt"))
                extra_time = extra_time + time.clock() - start_evaluate
                if i == 0:
                    log_file = open(log_filename, 'w')
                else:
                    log_file = open(log_filename, 'a')
                    #log_file.write(str(time.clock() - start_time - extra_time) + "," + str(test_accuracy) + "\n")
                    log_file.close()
            
            

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, y = mnist.train.next_batch(FLAGS.batch_size)
                if translated:
                    x = convertTranslated(x)
                
                if pretrain:
                
                    _, loss_value = sess.run([train_encoder, reconstruction_loss], {input_tensor: x, onehot_labels: y, learning_rate: FLAGS.g_learning_rate})
                    
                    encoder_loss += loss_value
                    
                    _, loss_value = sess.run([train_discrimator, D_loss], {input_tensor: x, onehot_labels: y, learning_rate: FLAGS.d_learning_rate})
                    discriminator_loss += loss_value

                    _, loss_value, imgs = sess.run([train_generator, G_loss, G], {input_tensor: x, onehot_labels: y, learning_rate: FLAGS.g_learning_rate})
                    generator_loss += loss_value
        
                else:
                    _, reward_value = sess.run([train_classifier, reward], {input_tensor: x, onehot_labels: y, learning_rate: FLAGS.d_learning_rate})
                    reward_fetched += reward_value

            discriminator_loss = discriminator_loss / FLAGS.updates_per_epoch
            generator_loss = generator_loss / FLAGS.updates_per_epoch
            encoder_loss = encoder_loss / FLAGS.updates_per_epoch
            reward_fetched = reward_fetched / FLAGS.updates_per_epoch

            print(switch)
            print("Enc. loss %f, Gen. loss %f, Disc. loss: %f, Reward: %f" % (encoder_loss, generator_loss,
                                                    discriminator_loss, reward_fetched))
                                                    

            if pretrain:
                for k in range(FLAGS.batch_size):
                    imgs_folder = os.path.join(FLAGS.working_directory, 'imgs_' + str(switch))
                    if not os.path.exists(imgs_folder):
                        os.makedirs(imgs_folder)

                    imsave(os.path.join(imgs_folder, '%d.png') % k,
                           imgs[k].reshape(28, 28))
                           
                           

