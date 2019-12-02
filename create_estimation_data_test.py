# This dataset is created to test the effect of position variance of blobs in estimation. 
import sys
import numpy as np
import random
from model_settings import img_height, img_width, min_edge, max_edge


def get_dims():
    """Get the dimensions of the ith blob."""

    width = 3
    height = 3

    return width, height

def generate_data(testing, min_blobs, max_blobs): # MT
    num_blobs = 2 # test two blobs  
    n_labels = 2  
    total = 10000 # total testing images
    train = np.zeros([total, img_height*img_width]) # input img size
    label = np.zeros([total, n_labels])
    blob_list = np.ones([total, n_labels, 2]) 
    img_count = 0
    
    nOfItem = total
    i = 0 # amount of images for each blob number

    while (i < nOfItem):
        img = np.zeros(img_height*img_width) # input img size
        num_count = 0 # amount of blobs in each image 
        used = np.zeros((num_blobs, 4)) # check overlapping
        width, height = get_dims()

        while num_count < num_blobs:
            index = 0
            if num_count == 0:
                cX = img_width/2 - 1
                cY = img_height/2 - 1
            else:
                cX = random.randint(1, 99-width) # top left corner
                cY = random.randint(1, 99-height)

            while index < num_count:
                if cX+width+1 <= used[index, 0] or used[index, 0]+1+used[index, 2] <= cX or used[index, 1]+1+used[index,3] <= cY or cY+height+1<=used[index,1]: # check for no overlapping blobs
                    index = index + 1
                else:
                    cX = random.randint(1, 99-width)
                    cY = random.randint(1, 99-height)
                    index = 0

            blob_list[img_count, num_count, 0] = cX + width/2
            blob_list[img_count, num_count, 1] = cY + height/2
            
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

    np.set_printoptions(threshold=np.nan)
    
    return train, label, blob_list
