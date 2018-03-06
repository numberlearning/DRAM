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
#from DRAMcopy14 import convertTranslated, classifications, input_tensor, count_tensor, target_tensor, batch_size, glimpses, z_size, dims, read_n 
#from DRAM_move_attn import viz_data, input_tensor, target_tensor, dims, read_n, glimpses, z_size
#from DRAM_classify_blobs import classification, classifications, x, batch_size, glimpses, z_size, dims, read_n
from DRAM_move_attn_0done import classifications, batch_size, glimpses, count_output_size, input_tensor, count_tensor, target_tensor 
#from DRAM_move_attn_sigmoid import classifications, count_output_size, input_tensor, count_tensor, target_tensor
#from DRAM_move_attn_sigmoid import classifications, count_output_size, input_tensor, count_tensor, target_tensor as classifications_sigmoid, count_output_size_sigmoid, input_tensor_sigmoid, count_tensor_sigmoid, target_tensor_sigmoid
#batch_size = 1
import load_input
import load_teacher

output_size = count_output_size
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


def random_count_image_sigmoid(num_imgs):
    """Get batch of random images from test set."""

    data = load_teacher.Teacher()
    data.get_test(even=1, done_vector='padding')
    batch_imgs, _, _, batch_counts, batch_traces = data.next_explode_batch(num_imgs)
    #i = random.randrange(len(batch_imgs))
    #return batch_imgs[i], batch_counts[i], batch_traces[i]
    return batch_imgs, batch_counts, batch_traces



def random_count_image(num_imgs):
    """Get batch of random images from test set."""

    data = load_teacher.Teacher()
    data.get_test(even=1, done_vector='end')
    batch_imgs, _, _, batch_counts, batch_traces = data.next_explode_batch(num_imgs)
    #i = random.randrange(len(batch_imgs))
    #return batch_imgs[i], batch_counts[i], batch_traces[i]
    return batch_imgs, batch_counts, batch_traces


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
    if path is None:
        path = "model_runs/DRAM_test_square"
    #path = "model_runs/sensical"
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))


def read_count_img(it, new_image):
    """Read image and visualize filterbanks."""

    batch_size = 1
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_count_image()
    img, counts, traces = last_image

    # dims [10, 100] => [1, 10, 100]
    imgs = np.expand_dims(imgs, axis=0)
    counts = np.expand_dims(counts, axis=0)
    traces = np.expand_dims(trace, axis=0)

    #feed_dict = { input_tensor: imgs } # testing doesn't work yet :(
    feed_dict = { input_tensor: imgs, count_tensor: counts, target_tensor: trace }

    #img = imgs[0][0]
    #flipped = np.flip(img.reshape(dims[0], dims[1]), 0)
    out = {
        #"img": flipped,
        "current_position": list(),
        "predict_position": list(),
        "target_position": list(),
        "count_tensors": list(), 
    }

    load_checkpoint(it)
    cs = sess.run(classifications, feed_dict=feed_dict)

    for i in range(len(cs)):
        # getting data from classifications at i-th glimpse of the first index
        current_x = list(cs[i]["current_x"])[0]
        current_y = list(cs[i]["current_y"])[0]
        predict_x = list(cs[i]["predict_x"])[0]
        predict_y = list(cs[i]["predict_y"])[0]
        target_x = list(cs[i]["target_x"])[0]
        target_y = list(cs[i]["target_y"])[0]
        predict_cnt = list(cs[i]["predict_cnt"])[0]
        target_cnt = list(cs[i]["target_cnt"])[0]
        print("current (x, y): ", current_x, ", ", current_y)
        print("predict (x, y): ", predict_x, ", ", predict_y)
        print("target (x, y): ", target_x, ", ", target_y)
        print("predict target cnt: ", predict_cnt, ", ", target_cnt)
        #out["dot"].append(dict(mu_x_list=predict_x, mu_y_list=predict_y))
        out["current_position"].append(dict(x=current_x, y=current_y))
        out["predict_position"].append(dict(x=predict_x, y=predict_y))
        out["target_position"].append(dict(x=target_x, y=target_y))
        out["predict_tensor"].append(dict(predict=predict_cnt, target=target_cnt))
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

    feed_dict = {input_tensor: imgs, target_tensor: poss }

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

def classify_count_imgs_sigmoid(it, new_image, num_imgs, path=None):
    #batch_size = 1
    print(new_image)
    print(path)
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_count_image_sigmoid(num_imgs)

    imgs, counts, traces = last_image
    imgs = np.asarray(imgs)
    print(imgs.shape)
    counts = np.asarray(counts)
    traces = np.asarray(traces)
    #imgs = np.expand_dims(imgs, axis=0)
    #counts = np.expand_dims(counts, axis=0)
    #traces = np.expand_dims(traces, axis=0)

    #imgs = np.asarray(imgs)
    out = list()
    
    load_checkpoint(it, human=False, path=path)
    feed_dict = {input_tensor: imgs, count_tensor: counts, target_tensor: traces }

    machine_cs  = sess.run(classifications, feed_dict)

    for idx in range(num_imgs):
        #all of these lists have glimpses number of items (across glimpses)
        current_x = list()
        current_y = list()

        predict_x = list()
        predict_y = list()

        target_x = list()
        target_y = list()

        predict_cnt = list()
        target_cnt = list()

        for i in range(len(machine_cs)):
            current_x.append(machine_cs[i]["current_x"][idx])
            current_y.append(machine_cs[i]["current_y"][idx])

            predict_x.append(machine_cs[i]["predict_x"][idx])
            predict_y.append(machine_cs[i]["predict_y"][idx])

            target_x.append(machine_cs[i]["target_x"][idx])
            target_y.append(machine_cs[i]["target_y"][idx])
            
            predict_cnt.append(machine_cs[i]["predict_cnt"][idx])
            target_cnt.append(machine_cs[i]["target_cnt"][idx])

        out.append({
            "current_x": current_x,
            "current_y": current_y,
            "predict_x": predict_x,
            "predict_y": predict_y,
            "target_x": target_x,
            "target_y": target_y,
            "predict_cnt": predict_cnt,
            "target_cnt": target_cnt,
        })

    return out

def classify_count_imgs(it, new_image, num_imgs, path=None):
    #batch_size = 1
    print(new_image)
    print(path)
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_count_image(num_imgs)

    imgs, counts, traces = last_image
    imgs = np.asarray(imgs)
    print(imgs.shape)
    counts = np.asarray(counts)
    traces = np.asarray(traces)
    #imgs = np.expand_dims(imgs, axis=0)
    #counts = np.expand_dims(counts, axis=0)
    #traces = np.expand_dims(traces, axis=0)

    #imgs = np.asarray(imgs)
    out = list()
    
    load_checkpoint(it, human=False, path=path)
    feed_dict = {input_tensor: imgs, count_tensor: counts, target_tensor: traces }

    machine_cs  = sess.run(classifications, feed_dict)

    for idx in range(num_imgs):
        #all of these lists have glimpses number of items (across glimpses)
        current_x = list()
        current_y = list()

        predict_x = list()
        predict_y = list()

        target_x = list()
        target_y = list()

        predict_cnt = list()
        target_cnt = list()

        for i in range(len(machine_cs)):
            current_x.append(machine_cs[i]["current_x"][idx])
            current_y.append(machine_cs[i]["current_y"][idx])

            predict_x.append(machine_cs[i]["predict_x"][idx])
            predict_y.append(machine_cs[i]["predict_y"][idx])

            target_x.append(machine_cs[i]["target_x"][idx])
            target_y.append(machine_cs[i]["target_y"][idx])
            
            predict_cnt.append(machine_cs[i]["predict_cnt"][idx])
            target_cnt.append(machine_cs[i]["target_cnt"][idx])

        out.append({
            "current_x": current_x,
            "current_y": current_y,
            "predict_x": predict_x,
            "predict_y": predict_y,
            "target_x": target_x,
            "target_y": target_y,
            "predict_cnt": predict_cnt,
            "target_cnt": target_cnt,
        })

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


def classify_image(it, new_image):
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

    load_checkpoint(it, human=False)
    human_cs = machine_cs = sess.run(classifications, feed_dict={x: imgs.reshape(batch_size, dims[0] * dims[1])})

    print(len(machine_cs)) # glimpses

    for i in range(len(machine_cs)):

        out["rs"].append((np.flip(machine_cs[i]["r"][0].reshape(read_n, read_n), 0), np.flip(human_cs[i]["r"][0].reshape(read_n, read_n), 0)))
        
        out["classifications"].append((machine_cs[i]["classification"][0], human_cs[i]["classification"][0]))

        stats_arr1 = np.asarray(machine_cs)
        stats_arr = stats_arr1[i]["stats"]
        
        out["rects"].append((stats_to_rect((machine_cs[i]["stats"][0][0], machine_cs[i]["stats"][1][0], machine_cs[i]["stats"][2][0])), stats_to_rect((human_cs[i]["stats"][0][0], human_cs[i]["stats"][1][0], human_cs[i]["stats"][2][0]))))

        gx, gy, delta = machine_cs[i]["stats"]
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

    minY = dims[0] - gy + read_n/2.0 * delta
    maxY = dims[0] - gy - read_n/2.0 * delta

    minX = gx - read_n/2.0 * delta
    maxX = gx + read_n/2.0 * delta
   
   # minX = gx - sum(delta[0:read_n//2])
   # maxX = gx + sum(delta[0:read_n//2])

   # minY = gy - sum(delta[0:read_n//2])
   # maxY = gy + sum(delta[0:read_n//2])


    if minX < 1:
        minX = 1

    if maxY < 1:
        maxY = 1

    if maxX > dims[0] - 1:
        maxX = dims[0] - 1

    if minY > dims[1] - 1:
        minY = dims[1] - 1

    return dict(top=[int(minY)], bottom=[int(maxY)], left=[int(minX)], right=[int(maxX)])


def list_to_dots(mu_x, mu_y, full_list=False):
    """Draw filterbank based on mu_x and mu_y."""

    if full_list:
        mu_x_list = np.reshape(mu_x, (1, read_n*read_n))
        mu_y_list = np.reshape(mu_y, (1, read_n*read_n))
    else:
        mu_x_list = mu_x * read_n
        mu_y_list = [val for val in mu_y for _ in range(0, read_n)]

    return dict(mu_x_list=mu_x_list, mu_y_list=mu_y_list)




print("analysis.py")
