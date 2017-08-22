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
#from DRAMcopy13 import convertTranslated, classification, classifications, x, batch_size, glimpses, z_size, dims, read_n 
#from DRAMcopy13_rewrite_filterbank import convertTranslated, classification, classifications, x, batch_size, glimpses, z_size, dims, read_n 
from DRAMcopy13_rewrite_filterbank3 import convertTranslated, classification, classifications, x, batch_size, glimpses, z_size, dims, read_n 
#from DRAMcopy14 import convertTranslated, classifications, input_tensor, count_tensor, target_tensor, batch_size, glimpses, z_size, dims, read_n 
#from DRAMcopy15 import viz_data, input_tensor, target_tensor, dims, read_n, glimpses, z_size
#from DRAMtest import classification, classifications, x, batch_size, glimpses, z_size, dims, read_n
#batch_size = 1
import load_input
import load_teacher

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
    x_test, y_test = data.next_batch(num_imgs)
    return x_test, y_test


def random_count_image():
    """Get batch of random images from test set."""

    data = load_teacher.Teacher()
    data.get_test(1)
    x_test, _, _, count_test, y_test = data.next_explode_batch(1)
    i = random.randrange(len(x_test))
    return x_test[i], count_test[i], y_test[i]


def random_image():
    """Get a transformed random image from test set."""

    batch_size = 1
    num_images = len(data.images)
    i = random.randrange(num_images)
    image_ar = np.array(data.images[i]).reshape((batch_size, dims[0], dims[1]))
    translated = convertTranslated(image_ar)
    return translated[0], data.labels[i]


def load_checkpoint(it, human=False, path=None):
    #path = "model_runs/regimen"
    #path = "model_runs/rewrite_filterbank"
    #path = "model_runs/DRAM_test_square"
    path = "model_runs/rewrite_filterbank3_test"
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))


def read_img(it, new_image):
    """Read image and visualize filterbanks."""

    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_count_image()
    imgs, _, poss = last_image

    # dims [10, 100] => [1, 10, 100]
    imgs = np.expand_dims(imgs, axis=0)
    poss = np.expand_dims(poss, axis=0)

    #feed_dict = { input_tensor: imgs } # testing doesn't work yet :(
    feed_dict = { input_tensor: imgs, target_tensor: poss }

    img = imgs[0][0]
    flipped = np.flip(img.reshape(dims[0], dims[1]), 0)
    out = {
        "img": flipped,
        "dots": list(),
        "dot": list(),
    }

    load_checkpoint(it)
    cs = sess.run(viz_data, feed_dict=feed_dict)

    for i in range(len(cs)):
        mu_x = list(cs[i]["mu_x"])
        mu_y = list(cs[i]["mu_y"])
        out["dots"].append(list_to_dots(mu_x, mu_y))

        predict_x = list(cs[i]["predict_x"])[0]
        predict_y = list(cs[i]["predict_y"])[0]
        print("prediction (x, y): ", predict_x, ", ", predict_y)
        out["dot"].append(dict(mu_x_list=predict_x, mu_y_list=predict_y))
    return out


def read_img2(it, new_image):
    """Read image and visualize filterbanks."""

    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_count_image()
    imgs, _, poss = last_image

    # dims [10, 100] => [1, 10, 100]
    imgs = np.expand_dims(imgs, axis=0)
    poss = np.expand_dims(poss, axis=0)

    feed_dict = { input_tensor: imgs, target_tensor: poss }

    img = imgs[0][0]
    flipped = np.flip(img.reshape(dims[0], dims[1]), 0)
    out = {
        "img": flipped,
        "dots": list(),
    }

    load_checkpoint(it)
    cs = sess.run(viz_data, feed_dict=feed_dict)

    for i in range(len(cs)):
        gx = list(cs[i]["gx"])[0]
        gy = list(cs[i]["gy"])[0]
        print("gx: ", cs[i]["gx"], ", gy: ", cs[i]["gy"])
        out["dots"].append(dict(mu_x_list=gx, mu_y_list=gy))

    return out


def classify_imgs2(it, new_imgs, num_imgs, path=None):
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = random_imgs(num_imgs)

    imgs, labels = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
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


def count_blobs(it, new_image):
    glimpses = 11
    global last_image
    if new_image or last_image is None:
        last_image = random_count_image()
    
    imgs, cnts, poss = last_image

    # dims [11, 100] => [1, 11, 100]
    imgs = np.expand_dims(imgs, axis=0)
    cnts = np.expand_dims(cnts, axis=0)
    poss = np.expand_dims(poss, axis=0)

    load_checkpoint(it, human=False)
    feed_dict = { input_tensor: imgs, count_tensor: cnts, target_tensor: poss }
    cs = sess.run(classifications, feed_dict=feed_dict)

    out = list()
    for g in range(glimpses):
        img = imgs[0][g]
        flipped = np.flip(img.reshape(10, 10), 0)
        item = {
            "img": flipped,
            "pos": cs[g]["position"],
            "cnt": cs[g]["count"],
        }
        out.append(item)
    return out


def classify_image(new_image):
    batch_size = 1#10000#100
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

    load_checkpoint(10000,human=False)
    human_cs = machine_cs = sess.run(classifications, feed_dict={x: imgs.reshape(batch_size, dims[0] * dims[1])})

    print(len(machine_cs)) # glimpses

    for i in range(len(machine_cs)):

        out["rs"].append((np.flip(machine_cs[i]["r"][0].reshape(read_n, read_n), 0), np.flip(human_cs[i]["r"][0].reshape(read_n, read_n), 0)))
        
        out["classifications"].append((machine_cs[i]["classification"][0], human_cs[i]["classification"][0]))

        stats_arr1 = np.asarray(machine_cs)
        stats_arr = stats_arr1[i]["stats"]
        
        out["rects"].append((stats_to_rect((machine_cs[i]["stats"][3][0], machine_cs[i]["stats"][4][0], machine_cs[i]["stats"][5][0])), stats_to_rect((human_cs[i]["stats"][3][0], human_cs[i]["stats"][4][0], human_cs[i]["stats"][5][0]))))

        _, _, _, gx, gy, delta = machine_cs[i]["stats"]
        print("gx: ", gx)
        print("gy: ", gy)
        print("delta: ", delta)
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
    """Draw attention window based on gx, gy, and delta."""

    gx, gy, delta = stats
    minX = gx - sum(delta[0:read_n//2])
    maxX = gx + sum(delta[0:read_n//2])

    minY = gy - sum(delta[0:read_n//2])
    maxY = gy + sum(delta[0:read_n//2])


    if minX < 1:
        minX = 1

    if maxY < 1:
        maxY = 1

    if maxX > dims[0] - 1:
        maxX = dims[0] - 1

    if minY > dims[1] - 1:
        minY = dims[1] - 1

    return dict(top=[int(minY)], bottom=[int(maxY)], left=[int(minX)], right=[int(maxX)])


def list_to_dots(mu_x, mu_y):
    """Draw filterbank based on mu_x and mu_y."""

    mu_x_list = mu_x * read_n
    mu_y_list = [val for val in mu_y for _ in range(0, read_n)]
 
    return dict(mu_x_list=mu_x_list, mu_y_list=mu_y_list)

print("analysis.py")
