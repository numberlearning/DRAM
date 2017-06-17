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
from tensorflow.python.platform import gfile
import tarfile
import cPickle
import gzip
from tensorflow.models.image.cifar10 import cifar10_input
import re


from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
from load_mnist import load_data
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

TOWER_NAME = 'tower'

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

translated = False
dims = [32, 32]
convdims = [16, 16]

img_size = dims[1]*dims[0]
read_n = 5
read_size = read_n*read_n
z_size=10
glimpses=10
enc_size = 256
dec_size = 256
train_iters = 1000000
batch_size = 100
eps = 1e-8
switch = 0
pretrain = True

#cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1
#cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
global xtest
global ytest
global train_batch
global test_batch
train_batch = 0
test_batch = 0


flags.DEFINE_integer("updates_per_epoch", 100, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000000, "max epoch")
flags.DEFINE_float("g_learning_rate", 1e-1, "learning rate")
flags.DEFINE_float("d_learning_rate", 1e-1, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_float("hidden_size", 10, "hidden size")
flags.DEFINE_float("gamma", 0.01, "gamma")

FLAGS = flags.FLAGS


dist_size = (9, 9)
ORG_SHP = [28, 28]
OUT_SHP = [100, 100]
NUM_DISTORTIONS_DB = 100000
mnist_data = load_data('mnist.pkl.gz')



def unpickle_cifar_dic(file):
    """
        Helper function: unpickles a dictionary (used for loading CIFAR)
        :param file: filename of the pickle
        :return: tuple of (images, labels)
        """
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def extract_cifar10(local_url='cifar-10-python.tar.gz', data_dir='cifar/'):
  """
  Extracts the CIFAR-10 dataset and return numpy arrays with the different sets
  :param local_url: where the tar.gz archive is located locally
  :param data_dir: where to extract the archive's file
  :return: a tuple (train data, train labels, test data, test labels)
  """
  # These numpy dumps can be reloaded to avoid performing the pre-processing
  # if they exist in the working directory.
  # Changing the order of this list will ruin the indices below.
  preprocessed_files = ['/cifar10_train.npy',
                        '/cifar10_train_labels.npy',
                        '/cifar10_test.npy',
                        '/cifar10_test_labels.npy']

  all_preprocessed = True
  for file in preprocessed_files:
    if not gfile.Exists(data_dir + file):
      all_preprocessed = False
      break

  if all_preprocessed:
    # Reload pre-processed training data from numpy dumps
    with gfile.Open(data_dir + preprocessed_files[0], mode='r') as file_obj:
      train_data = np.load(file_obj)
    with gfile.Open(data_dir + preprocessed_files[1], mode='r') as file_obj:
      train_labels = np.load(file_obj)

    # Reload pre-processed testing data from numpy dumps
    with gfile.Open(data_dir + preprocessed_files[2], mode='r') as file_obj:
      test_data = np.load(file_obj)
    with gfile.Open(data_dir + preprocessed_files[3], mode='r') as file_obj:
      test_labels = np.load(file_obj)

  else:
    # Do everything from scratch
    # Define lists of all files we should extract
    train_files = ["data_batch_" + str(i) for i in xrange(1,6)]
    test_file = ["test_batch"]
    cifar10_files = train_files + test_file

    # Check if all files have already been extracted
    need_to_unpack = False
    for file in cifar10_files:
      if not gfile.Exists(file):
        need_to_unpack = True
        break

    # We have to unpack the archive
    if need_to_unpack:
      tarfile.open(local_url, 'r:gz').extractall(data_dir)

    # Load training images and labels
    images = []
    labels = []
    for file in train_files:
      # Construct filename
      filename = data_dir + "/cifar-10-batches-py/" + file

      # Unpickle dictionary and extract images and labels
      images_tmp, labels_tmp = unpickle_cifar_dic(filename)

      # Append to lists
      images.append(images_tmp)
      labels.append(labels_tmp)

    # Convert to numpy arrays and reshape in the expected format
    train_data = np.asarray(images, dtype=np.float32).reshape((50000,3,32,32))
    train_data = np.swapaxes(train_data, 1, 3)
    train_labels = np.asarray(labels, dtype=np.int32).reshape(50000)

    # Save so we don't have to do this again
    np.save(data_dir + preprocessed_files[0], train_data)
    np.save(data_dir + preprocessed_files[1], train_labels)

    # Construct filename for test file
    filename = data_dir + "/cifar-10-batches-py/" + test_file[0]

    # Load test images and labels
    test_data, test_images = unpickle_cifar_dic(filename)

    # Convert to numpy arrays and reshape in the expected format
    test_data = np.asarray(test_data,dtype=np.float32).reshape((10000,3,32,32))
    test_data = np.swapaxes(test_data, 1, 3)
    test_labels = np.asarray(test_images, dtype=np.int32).reshape(10000)

    # Save so we don't have to do this again
    np.save(data_dir + preprocessed_files[2], test_data)
    np.save(data_dir + preprocessed_files[3], test_labels)

  return train_data, train_labels, test_data, test_labels



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
global READ_REUSE
REUSE=None
READ_REUSE = None



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
    a = tf.reshape(tf.cast(tf.range(convdims[0]), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(convdims[1]), tf.float32), [1, 1, -1])
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
    global READ_REUSE
    with tf.variable_scope(scope,reuse=READ_REUSE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(convdims[0]+1)/2*(gx_+1)
    gy=(convdims[1]+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(convdims[0],convdims[1])-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)


def read(x,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,convdims[0], convdims[1]])
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

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op



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
            reshape([batch_size, 32, 32, 3]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten())


def convattent(input_tensor):
    return (pt.wrap(input_tensor).
            reshape([batch_size, 32, 32, 3]).
            conv2d(5, 4, stride=2).dropout(0.9))
        
        #conv2d(5, 64, stride=2).
#conv2d(5, 128, edges='VALID').

            

def encoder(x):
    global REUSE
    global READ_REUSE
    print(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 1]).get_shape())

    x = convattent(x)

    print(x.get_shape())
    
    lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, read_size+dec_size) # encoder Op
    lstm_z = tf.nn.rnn_cell.LSTMCell(z_size, enc_size) # decoder Op
    
    outputs=[0] * glimpses
    h_z_prev=tf.zeros((batch_size,z_size))
    h_enc_prev=tf.zeros((batch_size,enc_size))
    enc_state=lstm_enc.zero_state(batch_size, tf.float32)
    z_state=lstm_z.zero_state(batch_size, tf.float32)
    
    
    for glimpse in range(glimpses):
        rs = []
        
        for i in range(4):
            rs.append(read(tf.slice(x, [0, 0, 0, i], [-1, -1, -1, 1]), h_z_prev))
            READ_REUSE = True
        rs.append(h_z_prev)
        with tf.variable_scope("encoder", reuse=REUSE):
            h_enc, enc_state = lstm_enc(tf.concat(1,rs), enc_state)
        
        
        with tf.variable_scope("z",reuse=REUSE):
            h_z=linear(h_enc,z_size)
        
        h_z_prev=h_z
        h_enc_prev=h_enc
        REUSE=True
    return h_z




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

    input_sample = tf.reshape(input_tensor, [batch_size, 1, 1, z_size])
    all_colors = []
    for i in range(3):
        with tf.variable_scope(str(i)):
            all_colors.append(tf.reshape((pt.wrap(input_sample).
                deconv2d(4, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor, [batch_size, dims[0], dims[1]]))
    return tf.pack(all_colors, axis=3)

    
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
    print('eval')
    global xtest
    global ytest
    global test_batch
    batches_in_epoch = 10000 // batch_size
    accuracy = 0
    
    for i in xrange(batches_in_epoch):
        xtest_cur = xtest[test_batch:test_batch+100]
        ytest_cur = ytest[test_batch:test_batch+100]
        test_batch = (test_batch + 100) % 10000
        
        
        
        feed_dict={input_tensor:xtest_cur, onehot_labels:dense_to_one_hot(ytest_cur), learning_rate:0}
        
        
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

    input_tensor = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))
    xtrain, ytrain, xtest, ytest = extract_cifar10()



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



                    
    reconstruction_loss = binary_crossentropy(tf.sigmoid(input_tensor), tf.sigmoid(G))
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

    train_pure = pt.apply_optimizer(optimizer, losses=[reconstruction_loss], regularize=True, include_marked=True)

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
        #saver.restore(sess, "drawmodel_cifarconv_gan_332_4000.ckpt")
        
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
            if (epoch % 10 == 0):
                log_filename = "convnet_gan_332_log_from_4000.csv"
                saver = tf.train.Saver(tf.all_variables())
                start_evaluate = time.clock()
                test_accuracy = evaluate()
                #print("Model saved in file: %s" % saver.save(sess, "classifymodel_cifarconv_gan_332_from_4000_" + str(100*epoch) + ".ckpt"))
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
                x = xtrain[train_batch:train_batch+100]
                y = dense_to_one_hot(ytrain[train_batch:train_batch+100])
                train_batch = (train_batch + 100) % 50000
                
                if pretrain:
                    
                    
                    #_, loss_value = sess.run([train_pure, reconstruction_loss], {input_tensor: x, onehot_labels: y, learning_rate: FLAGS.g_learning_rate})

#encoder_loss += loss_value
                    

                
                    _, loss_value = sess.run([train_encoder, reconstruction_loss], {input_tensor: x, onehot_labels: y, learning_rate: FLAGS.d_learning_rate})
                    
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
                                                    
'''
            if pretrain:
                for k in range(batch_size):
                    imgs_folder = os.path.join(FLAGS.working_directory, 'imgs_' + str(switch))
                    if not os.path.exists(imgs_folder):
                        os.makedirs(imgs_folder)

                    imsave(os.path.join(imgs_folder, '%d.png') % k,
                           imgs[k].reshape(28, 28))
                           
                           

'''
