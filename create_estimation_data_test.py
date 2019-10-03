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
    blob_list = np.zeros([total, 2]) 
    img_count = 0
    
    nOfItem = total
    i = 0 # amount of images for each blob number

    while (i < nOfItem):
        img = np.zeros(img_height*img_width) # input img size
        num_count = 0 # amount of blobs in each image 
        used = np.zeros((num_blobs, 4)) # check overlapping
        width, height = get_dims()

        # Place first blob
        cX1 = cX = int(img_width/2) - 1
        cY1 = cY = int(img_height/2) - 1
        spacing = 1

        for p in range(cY, cY+height):
            for q in range(cX, cX+width):
                img[p*img_width+q] = 255

        # Place second blob
        cX = random.randint(1, 99-width)
        cY = random.randint(1, 99-height)

        while not (cX+width+spacing <= cX1 or cX1+spacing+width <= cX or cY1+spacing+height <= cY or cY+height+spacing <= cY1): # check for no overlapping blobs
            cX = random.randint(1, 99-width)
            cY = random.randint(1, 99-height)

        blob_list[img_count][0] = cX + width/2
        blob_list[img_count][1] = cY + height/2

        for p in range(cY, cY+height):
            for q in range(cX, cX+width):
                img[p*img_width+q] = 255
        
        train[img_count] = img
        label[img_count, num_blobs - 1] = 1
        img_count += 1
        i += 1

    np.set_printoptions(threshold=np.nan)
    
    return train, label, blob_list
