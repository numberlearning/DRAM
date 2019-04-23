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
from FF_estimation import classification, classifications, x, batch_size, output_size, dims, read_n, delta_1 
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
    x_test, y_test = data.next_batch(num_imgs)
    return x_test, y_test # x_test: batch_imgs, y_test: batch_lbls

def load_checkpoint(it, human=False, path=None):
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))

def classify_imgs2(it, new_imgs, num_imgs, path=None): 
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

def classify_image(it, new_image):
    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_image()

    img, label = last_image
    imgs = np.zeros((batch_size, 100, 100))
    flipped = np.flip(img.reshape(100, 100), 0)
    imgs[0] = flipped

    out["img"] = flipped
    out["class"] = np.argmax(label)
    out["label"] = label
    out["classifications"] = list()
    out["rects"] = list()
    out["rs"] = list()
    out["h_decs"] = list()

    load_checkpoint(it, human=False)
    outer_cs = inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(batch_size, dims[0] * dims[1])})
    
    print(len(outer_cs)) # glimpses

    for i in range(len(outer_cs)):
        out["rs"].append((np.flip(outer_cs[i]["r_1"][0].reshape(read_n, read_n), 0), np.flip(inner_cs[i]["r_2"][0].reshape(read_n, read_n), 0)))
        out["classifications"].append((outer_cs[i]["classification"][0], inner_cs[i]["classification"][0]))

        stats_arr1 = np.asarray(machine_cs)
        stats_arr = stats_arr1[i]["stats"]
        
        out["rects"].append((stats_to_rect((outer_cs[i]["stats"][0][0], outer_cs[i]["stats"][1][0])), stats_to_rect((inner_cs[i]["stats"][0][0], inner_cs[i]["stats"][1][0]))))
        gx, gy = outer_cs[i]["stats"]
        print("gx: ", gx)
        print("gy: ", gy)

    return out

def stats_to_rect(stats):
    """Draw attention window based on gx, gy, and delta."""

    gx, gy = stats
    minX = sum(delta[0][0:read_n//2])
    maxX = 99

    minY = 2
    maxY = 99

    if minX < 1:
        minX = 1

    if maxY < 1:
        maxY = 1

    if maxX > dims[0] - 1:
        maxX = dims[0] - 1

    if minY > dims[1] - 1:
        minY = dims[1] - 1

    return dict(top=[int(minY)], bottom=[int(maxY)], left=[int(minX)], right=[int(maxX)])


print("analysis_twolayer.py")
