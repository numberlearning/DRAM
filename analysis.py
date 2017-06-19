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


from DRAMcopy10-nli_classification import convertTranslated, classification, classifications, x
from scipy.spatial.distance import cosine, euclidean
from sklearn.manifold import TSNE
from scipy.linalg import norm

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

data = mnist.input_data.read_data_sets("mnist", one_hot=True).test

def random_image():
    num_images = len(data.images)
    i = random.randrange(num_images)
    image_ar = np.array(data.images[i]).reshape((1, 28, 28))
    translated = convertTranslated(image_ar)
    return translated[0], data.labels[i]

def load_checkpoint(it, human):
    path = "attention-run" if human else "combined-run"
    saver.restore(sess, "%s/save_%d.ckpt" % (path, it))

last_image = None

def state_to_cell_array(classifications, key):
    out = np.zeros((len(classifications), 256))
    for i, d in enumerate(classifications):
        c = d[key].c
        out[i] = c[0]
    return out

def print_distances(ar, label):
    ds = list()
    for i in range(1, ar.shape[0]):
        ds.append(cosine(ar[i], ar[i - 1]))
    print label, ds

def do_tsne(ar):
    X = TSNE(n_components=2).fit_transform(ar)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    return X


def classify_image(it, new_image):
    out = dict()
    global last_image
    if new_image or last_image is None:
        last_image = random_image()

    img, label = last_image
    flipped = np.flip(img.reshape(100, 100), 0)

    out["img"] = flipped
    out["class"] = np.argmax(label)
    out["label"] = label
    out["classifications"] = list()
    out["rects"] = list()
    out["rs"] = list()



    load_checkpoint(it, human=False)
    machine_cs = sess.run(classifications, feed_dict={x: img.reshape(1, 10000)})

    # print_distances(state_to_cell_array(machine_cs, "enc_state"), "machine enc")
    # print_distances(state_to_cell_array(machine_cs, "dec_state"), "machine dec")

    # for d in machine_cs:
    #     i, f, o = d["dec_gates"]
    #     print i.shape, f.shape, o.shape

    #     print "MACHINE DEC I F O NORMS", map(np.mean, map(lambda x: x[0], d["dec_gates"]))

    # last = None

    # for d in machine_cs:
    #     dec_state = d["dec_state"]
    #     c = dec_state.c
    #     if last is not None:
    #         print "MACHINE DIST FROM LAST", euclidean(c, last)
    #     last = c
    #     # cell = dec_state[0]
    #     # print cell.shape


    load_checkpoint(it, human=True)
    human_cs = sess.run(classifications, feed_dict={x: img.reshape(1, 10000)})

    # print_distances(state_to_cell_array(human_cs, "enc_state"), "human enc")
    # print_distances(state_to_cell_array(human_cs, "dec_state"), "human dec")



    # h_last = None 
    # human_cells
    # for d in human_cs:
    #     dec_state = d["dec_state"]
    #     c = dec_state.c
    #     if h_last is not None:
    #         print "HUMAN DIST FROM LAST", euclidean(c, h_last)
    #     h_last = c

    for i in range(len(machine_cs)):
        out["rs"].append((np.flip(machine_cs[i]["r"].reshape(12, 12), 0), np.flip(human_cs[i]["r"].reshape(12, 12), 0)))
        out["classifications"].append((machine_cs[i]["classification"], human_cs[i]["classification"]))
        out["rects"].append((stats_to_rect(machine_cs[i]["stats"]), stats_to_rect(human_cs[i]["stats"])))


    # machine_cs = state_to_cell_array(machine_cs, "dec_state")
    # human_cs = state_to_cell_array(human_cs, "dec_state")

    # print np.array(do_tsne(machine_cs))
    # print np.array(do_tsne(human_cs))

    return out

def accuracy_stats(it, human):
    load_checkpoint(it, human)
    bsize = 1
    batches_in_epoch = len(data._images) // bsize
    accuracy = np.zeros(10)
    confidence = np.zeros(10)

    confusion = np.zeros((10, 10))

    print("STARTING", batches_in_epoch)
    
    for i in xrange(batches_in_epoch):
        nextX, nextY = data.next_batch(bsize)
        nextX = convertTranslated(nextX)
        cs = sess.run(classifications, feed_dict={x: nextX})

        y = nextY.reshape(10)
        l = np.argmax(y)
        # print cs[-1].shape, nextY.shape
        for j in range(10):
            c = cs[j]["classification"].reshape(10)
            p = np.argmax(c)
            # print c, nextY

            # c = c.reshape(10)
            accuracy[j] += 1 if p == l else 0
            confidence[j] += c[l]
            confusion[l, p] += 1
        if i % 1000 == 0:
            print(i, batches_in_epoch)
    
    accuracy /= float(batches_in_epoch)
    confidence /= float(batches_in_epoch)

    # return accuracy
    return confusion


def stats_to_rect(stats):
    Fx, Fy, gamma = stats
    
    def min_max(ar):
        minI = None
        maxI = None
        for i in range(100):
            if np.any(ar[0, :, i]):
                minI = i
                break
                
        for i in reversed(range(100)):
            if np.any(ar[0, :, i]):
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



# classify_image(300000, new_image=True)
print "ALL-STEP", accuracy_stats(300000, True)
print "LAST-STEP", accuracy_stats(300000, False)
