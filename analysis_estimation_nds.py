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
#from FF_estimation_scalar_swap import classification, classifications, x, batch_size, output_size, dims, read_n, delta_1, hidden_w, hidden_b
#print('FF_estimation_scalar_swap')
from FF_estimation_classifier_swap import classification, classifications, x, batch_size, output_size, dims, read_n, delta_1, hidden_w, hidden_b
print('FF_estimation_classifier_swap')
import load_input, load_estimation_test, load_incr_test

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

#saver_hidden = tf.train.Saver(["hidden/a", "hidden/b"])
#saver_output = tf.train.Saver()
saver = tf.train.Saver()
last_imgs = None

#data = load_input.InputData()
#data.get_test(1,9)

def random_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_input.InputData()
    data.get_test(1,min_blobs_test,max_blobs_test)
    x_test, y_test = data.next_batch_nds(num_imgs)
    return x_test, y_test # x_test: batch_imgs, y_test: batch_lbls

def training_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_input.InputData()
    data.get_train(0,min_blobs_test,max_blobs_test)
    x_train, y_train = data.next_batch_nds(num_imgs)
    return x_train, y_train # x_train: batch_imgs, y_train: batch_lbls

def density_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_input.InputData()
    data.get_density(0,min_blobs_test,max_blobs_test)
    x_train, y_train = data.next_batch_nds(num_imgs)
    return x_train, y_train # x_train: batch_imgs, y_train: batch_lbls

def CTA_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_input.InputData()
    data.get_CTA(0,min_blobs_test,max_blobs_test)
    x_train, y_train = data.next_batch_nds(num_imgs)
    return x_train, y_train # x_train: batch_imgs, y_train: batch_lbls

def has_spacing_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_input.InputData()
    data.get_has_spacing(0,min_blobs_test,max_blobs_test)
    x_train, y_train = data.next_batch_nds(num_imgs)
    return x_train, y_train # x_train: batch_imgs, y_train: batch_lbls

def one_fixed_imgs(num_imgs):
    """Get batch of random images from test set."""
    data = load_estimation_test.InputData()
    data.get_test(0,min_blobs_test,max_blobs_test)
    x_train, y_train, blts = data.next_batch(num_imgs)
    return x_train, y_train, blts # x_train: batch_imgs, y_train: batch_lbls

def incr_imgs():
    """Get batch of random images from test set."""
    data = load_incr_test.InputData()
    data.get_test(min_blobs_test, max_blobs_test)
    x_train, x_incr_train = data.next_batch()
    return x_train, x_incr_train

def scalar_imgs(num_imgs):
    """Get batch of random images from test set with scalar label."""
    data = load_input.InputData()
    data.get_test(1, min_blobs_test, max_blobs_test)
    x_test, y_test, _ = data.next_batch(num_imgs)
    return x_test, y_test

def po_imgs(num_imgs, incremental=False):
    """Get batch of random images from po test set."""
    data = load_input.InputData()
    data.load_po(incremental)
    x_test, y_test_scalar, y_test_classifier = data.next_batch_po(num_imgs)
    return x_test, y_test_scalar, y_test_classifier

def po_img_sets():
    """Get num_sets sets of images from po test set."""
    data = load_input.InputData()
    data.load_po(incremental=True)
    x_test, y_test_scalar, y_test_classifier = data.get_sets()
    return x_test, y_test_scalar, y_test_classifier

def split_imgs():
    """Get all the images from test set."""
    data = load_input.InputData()
    data.get_test(1,min_blobs_test,max_blobs_test)
    x1_test, y1_test, x2_test, y2_test = data.split_data()
    #print(np.asarray(x1_test).shape)
    #print(np.asarray(x2_test).shape)
    return x1_test, y1_test, x2_test, y2_test # x_test: batch_imgs, y_test: batch_lbls

def load_checkpoint(it, human=False, path=None):
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))

def load_checkpoint_swap(it, mode, test_run, human=False, path=None, scalar=False):
    swap_index = 3000000
    if scalar:
        swap_file = "model_runs/estimation/classifier_model/classifier_"+mode+"_run"+str(test_run+1)+"/classifymodel_" + str(swap_index) + ".ckpt"
    else:
        swap_file = "model_runs/estimation/scalar_model/scalar_"+mode+"_run"+str(test_run+1)+"/classifymodel_" + str(swap_index) + ".ckpt"
    ckpt_reader = tf.train.load_checkpoint(swap_file)
    loaded_hidden_w = ckpt_reader.get_tensor("hidden/w")
    loaded_hidden_b = ckpt_reader.get_tensor("hidden/b")
    checkpoint_path = "%s/classifymodel_%d.ckpt" % (path, it)
    #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    #print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False, all_tensor_names=False)
    #print(tf.all_variables())
    saver.restore(sess, checkpoint_path)
    return loaded_hidden_w, loaded_hidden_b

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

def classify_imgs_density(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = density_imgs(num_imgs)

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

def classify_imgs_CTA(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = CTA_imgs(num_imgs)

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

def classify_imgs_has_spacing(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = has_spacing_imgs(num_imgs)

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

def classify_imgs_one_fixed(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = one_fixed_imgs(num_imgs)

    imgs, labels, blob_coords = last_imgs
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
            "classifications": cs,
            "blob_coords": blob_coords
        }
        out.append(item)
    return out

def classify_imgs_incr(it, new_imgs, path=None): 
    out_all_N = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = incr_imgs()

    num_imgs = 100
    num_incr = 100
    imgs_all_N, imgs_incr_all_N = last_imgs
    load_checkpoint(it, human=False, path=path)

    for n, imgs in enumerate(imgs_all_N):
        N = n+1
        imgs = np.asarray(imgs)
        imgs_incr = imgs_incr_all_N[n]
        out = list()

        outer_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})

        for i, img in enumerate(imgs):

            flipped = np.flip(img.reshape(100, 100), 0)
            cs = list()
            cs.append(outer_cs[0]["classification"][i])

            cs_incr_list = list()
            flipped_incr_list = list()
            imgs_incr_for_img = imgs_incr[i]
            imgs_incr_for_img = np.asarray(imgs_incr_for_img)
            inner_cs = sess.run(classifications, feed_dict={x: imgs_incr_for_img.reshape(num_incr, dims[0] * dims[1])})
            for j, img_incr in enumerate(imgs_incr_for_img):
                flipped_incr = np.flip(img_incr.reshape(100,100),0)
                flipped_incr_list.append(flipped_incr)
                cs_incr = list()
                cs_incr.append(inner_cs[0]["classification"][j])
                cs_incr_list.append(cs_incr)

            item = {
                "img": flipped,
                "imgs_incr": flipped_incr_list,
                "N": N,
                "i": i,
                "classifications": cs,
                "classifications_incr": cs_incr_list
            }
            out.append(item)
        out_all_N.append(out)
    return out_all_N

def classify_imgs_scalar(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = scalar_imgs(num_imgs)

    imgs, labels = last_imgs
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})
    for idx in range(num_imgs):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append(inner_cs[0]["classification"][idx])

        item = {
            "img": flipped,
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)
    return out


def classify_imgs_po(it, new_imgs, num_imgs, path=None, incremental=False, scalar=False): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = po_imgs(num_imgs)

    imgs, labels_scalar, labels_classifier = last_imgs
    if scalar:
        labels = labels_scalar
    else:
        labels = labels_classifier
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})
    for idx in range(num_imgs):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append(inner_cs[0]["classification"][idx])

        item = {
            "img": flipped,
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)

    return out


def classify_imgs_po_swap(it, new_imgs, num_imgs, mode, test_run, path=None, incremental=False, scalar=False): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = po_imgs(num_imgs)

    imgs, labels_scalar, labels_classifier = last_imgs
    if scalar:
        labels = labels_scalar
    else:
        labels = labels_classifier
    imgs = np.asarray(imgs)

    loaded_hidden_w, loaded_hidden_b = load_checkpoint_swap(it, mode, test_run, human=False, path=path, scalar=scalar)
    inner_cs = sess.run(classifications,feed_dict={
                x: imgs.reshape(num_imgs, dims[0] * dims[1]),
                hidden_w: loaded_hidden_w,
                hidden_b: loaded_hidden_b})
    for idx in range(num_imgs):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append(inner_cs[0]["classification"][idx])

        item = {
            "img": flipped,
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)

    return out


def classify_imgs_po_sets(it, new_imgs, path=None, scalar=False): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = po_img_sets()

    imgs, labels_scalar, labels_classifier = last_imgs
    if scalar:
        labels = labels_scalar
    else:
        labels = labels_classifier
    imgs = np.asarray(imgs)

    load_checkpoint(it, human=False, path=path)
    num_sets = 1000
    num_numerosities = 9
    num_imgs = num_sets * num_numerosities
    inner_cs = sess.run(classifications, feed_dict={x: imgs.reshape(num_imgs, dims[0] * dims[1])})
    for idx in range(num_imgs):
        img = imgs[idx]
        flipped = np.flip(img.reshape(100, 100), 0)
        cs = list()
        cs.append(inner_cs[0]["classification"][idx])

        item = {
            "img": flipped,
            "label": labels[idx],
            "classifications": cs
        }
        out.append(item)

    return out


def classify_imgs_training(it, new_imgs, num_imgs, path=None): 
    out = list()
    global last_imgs
    if new_imgs or last_imgs is None:
        last_imgs = training_imgs(num_imgs)

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
    #print(imgs.shape)

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
sess.close()
print("analysis_estimation_nds.py")
