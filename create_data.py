# MT: modifications by Mengting

import sys
import numpy as np
import random
from model_settings import img_height, img_width, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

def set_size(num_blobs, even=None):
    """Get amount of images for each number""" 
    if even is None:
        return int(10000/(num_blobs**2)) # uneven: number distribution 1/(n^2)
    else:
        return 1000 # even: make 1000 images for each number

def get_total(even, min_blobs, max_blobs): # MT
    """Get total amount of images."""
    c = min_blobs
    total = 0
    while (c < max_blobs + 1):
       total = total + set_size(c, even)
       c = c + 1
    return total

def generate_data(even, min_blobs, max_blobs): # MT
    n_labels = max_blobs_train - min_blobs_train + 1 
    total = get_total(even, min_blobs, max_blobs)
    train = np.zeros([total, img_height*img_width]) # input img size
    label = np.zeros([total, n_labels])
    num_blobs = min_blobs
    img_count = 0
        
    while (num_blobs < max_blobs + 1):

        nOfItem = set_size(num_blobs, even) # even, 1000; uneven, 10000/(num_blobs**2)
        i = 0 # amount of images for each blob number

        while (i < nOfItem):
            img = np.zeros(img_height*img_width) # input img size
            num_count = 0 # amount of blobs in each image 
            used = np.zeros((num_blobs, 4)) # check overlapping

            while num_count < num_blobs: 
                height = random.randint(min_edge, max_edge)
                #width = random.randint(min_edge, max_edge)
                width = height # for square blobs
                #cX = random.randint(1 * num_blobs * 10, 50-width + num_blobs * 10)
                #cY = random.randint(1 * num_blobs * 10, 50-height + num_blobs * 10)
                cX = random.randint(1, 99-width) # top left corner
                cY = random.randint(1, 99-height)
                
                index = 0
                
                while index < num_count:
                    if cX+width+1 <= used[index, 0] or used[index, 0]+1+used[index, 2] <= cX or used[index, 1]+1+used[index,3] <= cY or cY+height+1<=used[index,1]:
                        index = index + 1
                    else:
                        cX = random.randint(1, 99-width)
                        cY = random.randint(1, 99-height)
                        index = 0
    
                used[index, 0] = cX
                used[index, 1] = cY
                used[index, 2] = width
                used[index, 3] = height

                for p in range(cY, cY+height):
                    for q in range(cX, cX+width): 
                        img[p*img_width+q] = 255
                num_count += 1
          
            train[img_count] = img
            label[img_count, num_blobs - 1] = 1
            img_count += 1
            i += 1
          
        num_blobs += 1
    
    np.set_printoptions(threshold=np.nan)

#    if empty_img:
#        train = np.vstack((train, np.zeros((1000, 10000))))
#        for empty_idx in range(1000):
#            empty_label = np.zeros(max_blobs + 1)
#            empty_label[0] = 1
#            label = np.vstack((label, empty_label))

    return train, label
