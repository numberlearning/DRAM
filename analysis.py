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
from DRAMcopy10_nli_classification import convertTranslated, classification, classifications, x, batch_size, glimpses, z_size, dims, read_n 
import load_input

output_size = z_size
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

data = load_input.InputData()
data.get_test(1)


def random_image():
    num_images = len(data.images)
    i = random.randrange(num_images)
    image_ar = np.array(data.images[i]).reshape((1, dims[0], dims[1]))
    translated = convertTranslated(image_ar)
    return translated[0], data.labels[i]


def load_checkpoint(it, human):
    path = "model_runs/test_h_dec"
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))


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
    out["centers"] = list()
    out["h_decs"] = list()



    load_checkpoint(it, human=False)
    machine_cs = sess.run(classifications, feed_dict={x: img.reshape(1, dims[0] * dims[1])})

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
    human_cs = sess.run(classifications, feed_dict={x: img.reshape(batch_size, dims[0] * dims[1])})

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
        out["rs"].append((np.flip(machine_cs[i]["r"].reshape(read_n, read_n), 0), np.flip(human_cs[i]["r"].reshape(read_n, read_n), 0)))
        out["classifications"].append((machine_cs[i]["classification"], human_cs[i]["classification"]))
        out["rects"].append((stats_to_rect(machine_cs[i]["stats"]), stats_to_rect(human_cs[i]["stats"])))
        out["centers"].append((machine_cs[i]["more_stats"], human_cs[i]["more_stats"]))
        out["h_decs"].append((machine_cs[i]["h_dec"], human_cs[i]["h_dec"]))

    print(out["rects"])
    print(out["centers"])
    print(out["h_decs"])

    # machine_cs = state_to_cell_array(machine_cs, "dec_state")
    # human_cs = state_to_cell_array(human_cs, "dec_state")

    # print np.array(do_tsne(machine_cs))
    # print np.array(do_tsne(human_cs))

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

#         for glimpse in range(glimpses):
            c = cs[glimpses - 1]["classification"].reshape(batch_size, output_size)
            
            img_c = c[img]
            pred = np.argmax(img_c)

#             for img in range(batch_size):
#                 img_c = c[img]
#                 pred = np.argmax(img_c)
#                 accuracy[glimpse] += 1 if pred == label else 0
#                 confidence[glimpse] += img_c[label]
#                 confusion[label, pred] += 1
#             label = labels[img, 0]

            label = int(labels[img][0])
            pred_distr_at_glimpses[glimpses - 1, label, pred + 1] += 1
#             class_distr_at_glimpses[glimpses - 1, label, glimpse] = img_c[glimpse]  
        if i % 1000 == 0:
            print(i, batches_in_epoch)
    
    
#     accuracy /= float(batches_in_epoch)
#     confidence /= float(batches_in_epoch)
    return pred_distr_at_glimpses# , class_distr_at_glimpses


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



print("analysis.py")
