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
from DRAMcopy13 import convertTranslated, classification, classifications, x, batch_size, glimpses, z_size, dims, read_n 
#batch_size = 1
import load_input

output_size = z_size
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

data = load_input.InputData()
data.get_test(1)


def random_imgs(num_imgs):
    """Get batch of random images from test set."""

    data = load_input.InputData()
    data.get_test(1)
    data.next_batch(num_imgs)
    x_test, y_test = data.next_batch(num_imgs)
    return x_test, y_test


def random_image():
    """Get a transformed random image from test set."""

    batch_size = 1
    num_images = len(data.images)
    i = random.randrange(num_images)
    image_ar = np.array(data.images[i]).reshape((batch_size, dims[0], dims[1]))
    translated = convertTranslated(image_ar)
    return translated[0], data.labels[i]


def load_checkpoint(it, human):
    path = "model_runs/lined_up"
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))


def classify_imgs(it, new_imgs, num_imgs):
    out_imgs = list()
    for i in range(num_imgs):
        out_imgs.append(classify_image(it, new_imgs))
    return out_imgs


def classify_imgs2(it, new_imgs, num_imgs):
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = random_imgs(num_imgs)

    imgs, labels = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False)
    human_cs = machine_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})
    for idx in range(num_imgs):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        for i in range(len(machine_cs)):
            cs.append((machine_cs[i]["classification"][idx], human_cs[i]["classification"][idx]))

        item = {
            "img": flipped,
            "class": np.argmax(labels[idx]),
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)
    return out


def classify_image(it, new_image):
    batch_size = 10000
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
    human_cs = machine_cs = sess.run(classifications, feed_dict={x: imgs.reshape(batch_size, dims[0] * dims[1])})

    for i in range(len(machine_cs)):

        out["rs"].append((np.flip(machine_cs[i]["r"][0].reshape(read_n, read_n), 0), np.flip(human_cs[i]["r"][0].reshape(read_n, read_n), 0)))
        
        out["classifications"].append((machine_cs[i]["classification"][0], human_cs[i]["classification"][0]))

        stats_arr1 = np.asarray(machine_cs)
        stats_arr = stats_arr1[i]["stats"]
        
        out["rects"].append((stats_to_rect((machine_cs[i]["stats"][0][0], machine_cs[i]["stats"][1][0], machine_cs[i]["stats"][2][0])), stats_to_rect((human_cs[i]["stats"][0][0], human_cs[i]["stats"][1][0], human_cs[i]["stats"][2][0]))))

        out["h_decs"].append((machine_cs[i]["h_dec"][0], human_cs[i]["h_dec"][0]))


    return out



def accuracy_stats(it, human):
    load_checkpoint(it, human)
    batches_in_epoch = len(data.images) // batch_size
    accuracy = np.zeros(glimpses)
    confidence = np.zeros(glimpses)
    confusion = np.zeros((output_size + 1, output_size + 1))
    pred_distr_at_glimpses = np.zeros((glimpses, output_size, output_size + 1)) # 10x9x10
#     class_distr_at_glimpses = np.zeros((glimpses, output_size, output_size + 1))# 10x9x10

    print("STARTING, batches_in_epoch: ", batches_in_epoch)
    for i in range(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        cs = sess.run(classifications, feed_dict={x: nextX})

        y = np.asarray(nextY).reshape(batch_size, output_size)
        labels = np.zeros((batch_size, 1))
        for img in range(batch_size):
            labels[img] = np.argmax(y[img])

            c = cs[glimpses - 1]["classification"].reshape(batch_size, output_size)
            
            img_c = c[img]
            pred = np.argmax(img_c)

            label = int(labels[img][0])
            pred_distr_at_glimpses[glimpses - 1, label, pred + 1] += 1
        if i % 1000 == 0:
            print(i, batches_in_epoch)
    
    return pred_distr_at_glimpses# , class_distr_at_glimpses


def stats_to_rect(stats):
    Fx, Fy, gamma = stats
    
    def min_max(ar):
        minI = None
        maxI = None
        for i in range(100):
            if np.any(ar[:, i]):
                minI = i
                break
                
        for i in reversed(range(100)):
            if np.any(ar[:, i]):
                maxI = i
                break
                
        return minI, maxI

    minX, maxX = min_max(Fx)
    minY, maxY = min_max(Fy)
    
    if minX == 0:
        minX = 1
        
    if minY == 0:
        minY = 1
        
    if maxX == 100:
        maxX = 99
        
    if maxY == 100:
        maxY = 99
    
    return dict(
        top=[minY],
        bottom=[maxY],
        left=[minX],
        right=[maxX]
    )



print("analysis.py")
