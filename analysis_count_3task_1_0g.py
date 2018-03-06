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
from model_settings import min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test, glimpses
#from COUNT_viz_nogamma import classification, classifications, corrects, x, batch_size, output_size, dims, read_n, delta_1, delta_2 
#from COUNT_viz import classification, classifications, corrects, x, batch_size, output_size, dims, read_n, delta_1, delta_2 
from Ct_2l_3t_0g_1_viz import corrects, classification, classifications, points, x, count_word, blob_list, size_list, batch_size, output_size, dims, read_n, delta_1, delta_2 
#from COUNT_viz_onetask_nogamma import classification, classifications, corrects, x, batch_size, output_size, dims, read_n, delta_1, delta_2 
import load_count

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

def random_imgs(num_imgs):
    """Get batch of random images from test set."""

    data = load_count.InputData()
    #data.get_test(1,min_blobs_test,max_blobs_test)
    data.get_blank() 
    imgs_test, lbls_test, blts_test, slts_test, mlts_test, nlts_test, cwds_test = data.next_batch(num_imgs)
    return imgs_test, lbls_test, blts_test, slts_test, mlts_test, nlts_test, cwds_test

def load_checkpoint(it, human=False, path=None):
    saver.restore(sess, "%s/countmodel_%d.ckpt" % (path, it))

def classify_imgs2(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = random_imgs(num_imgs)

    imgs, lbls, blts, slts, mlts, nlts, cwds = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    #human_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})
    for idx in range(num_imgs):
        img = imgs[idx]
        cwd = cwds[idx]
        blt = blts[idx]
        slt = slts[idx]

        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        xs = list()
        ys = list()
        bs = list()

        human_cs = sess.run([classifications, points, corrects], feed_dict={x: img.reshape(batch_size, dims[0]*dims[1]), count_word: cwd.reshape(batch_size, glimpses, output_size+1), blob_list: blt.reshape(batch_size, glimpses, 2), size_list: slt.reshape(batch_size, glimpses)})
        #human_cs = sess.run(classifications, feed_dict={x: img.reshape(1, dims[0]*dims[1])})
        for glimpse in range(glimpses):
            cs.append(human_cs[0][glimpse+1]["classification"])
            xs.append(human_cs[1][glimpse]["gx"])
            ys.append(human_cs[1][glimpse]["gy"])
            bs.append(human_cs[1][glimpse]["blb_pot"])

        item = {
            "img": flipped,
            "class": np.argmax(lbls[idx]+1),
            "label": lbls[idx],
            "corrects": human_cs[2],
            "num": nlts[idx],
            "classifications": cs,
            "xs": xs,
            "ys": ys,
            "blob_point": bs,
        }
        out.append(item)
    return out

print("analysis_count.py")
