import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
from model_settings import min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test
from DRAMcopy13_onelayer import convertTranslated, classification, classifications, x, batch_size, output_size, dims, read_n, delta_1 
import load_input

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

#data = load_input.InputData()
#data.get_test(1,9)

def random_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_input.InputData()
    data.get_test(1,min_blobs_test,max_blobs_test)
    x_test, y_test = data.next_batch_nds(num_imgs)
    return x_test, y_test # x_test: batch_imgs, y_test: batch_lbls

def split_imgs():
    """Get all the images from test set."""
    data = load_input.InputData()
    data.get_test(1,min_blobs_test,max_blobs_test)
    x1_test, y1_test, x2_test, y2_test = data.split_data()
    return x1_test, y1_test, x2_test, y2_test # x_test: batch_imgs, y_test: batch_lbls

def load_checkpoint(it, human=False, path=None):
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))

def classify_imgs(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = random_imgs(num_imgs)

    imgs, labels = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    outer_cs = inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})
    for idx in range(num_imgs):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append((outer_cs[0]["classification"][idx], inner_cs[0]["classification"][idx]))

        item = {
            "img": flipped,
            "class": np.argmax(labels[idx]),
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)
    return out

def classify_imgs_fh(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = split_imgs()

    imgs, labels, _, _ = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    outer_cs = inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs//2, dims[0] * dims[1])})
    for idx in range(num_imgs//2):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append((outer_cs[0]["classification"][idx], inner_cs[0]["classification"][idx]))

        item = {
            "img": flipped,
            "class": np.argmax(labels[idx]),
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)
    return out

def classify_imgs_lh(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = split_imgs()

    _, _, imgs, labels = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    outer_cs = inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs//2, dims[0] * dims[1])})
    for idx in range(num_imgs//2):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append((outer_cs[0]["classification"][idx], inner_cs[0]["classification"][idx]))

        item = {
            "img": flipped,
            "class": np.argmax(labels[idx]),
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)
    return out

print("analysis_onelayer_nds.py")
