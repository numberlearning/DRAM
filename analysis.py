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
from DRAMcopy10_nli_classification import convertTranslated, classification, classifications, x, batch_size, glimpses, z_size 
import load_input

output_size = z_size
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

saver = tf.train.Saver()

data = load_input.InputData()
data.get_test(1)

def load_checkpoint(it, human):
    path = "model_runs/number_learning_new_prop"
    saver.restore(sess, "%s/classifymodel_%d.ckpt" % (path, it))


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

print("analysis.py")
