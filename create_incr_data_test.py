# This dataset is created to see the effect of incrementing the number of blobs in estimation.
import sys
import numpy as np
import random
import copy
from model_settings import img_height, img_width

num_imgs = 100 # total testing images with N blobs
num_incr = 100 # number of images with N+1 blobs per image with N blobs
margin = 10 # minimum number of pixels between blob and edge of image
spacing = 20 # minimum number of pixels between blobs
width = 3
height = 3
max_N = 9


def get_dims():
    """Get the dimensions of the ith blob."""

    return width, height


def generate_data():
    """Generate images for all numerosities."""
    train_all_N = np.zeros([max_N, num_imgs, img_height*img_width])
    train_incr_all_N = np.zeros([max_N, num_imgs, num_incr, img_height*img_width])
    for i in range(max_N):
        N = i+1
        print('numerosity in generate data: %d' % N)
        train, train_incr = generate_data_with_N_blobs(N)
        train_all_N[i] = train
        train_incr_all_N[i] = train_incr
        print('done with numerosity in generate data: %d' % N)
    return train_all_N, train_incr_all_N



def generate_data_with_N_blobs(N): # MT
    """Generate images with N blobs, and for each image, generate images with N+1 blobs."""

    num_blobs = N # test two blobs  
    num_blobs_incr = N+1
    train = np.zeros([num_imgs, img_height*img_width]) # input img size
    train_incr = np.zeros([num_imgs, num_incr, img_height*img_width])
    img_count = 0
    
    i = 0

    while (i < num_imgs):
        img = np.zeros(img_height*img_width) # input img size
        blob_idx = 0 # amount of blobs in each image 
        used = np.zeros((num_blobs, 4)) # check overlapping
        width, height = get_dims()

        while blob_idx < num_blobs + 1:
            index = 0

            cX = random.randint(margin, img_width-margin-width)
            cY = random.randint(margin, img_height-margin-height)

            tries = 0

            while index < blob_idx:
                if cX+width+spacing <= used[index, 0] or used[index, 0]+spacing+used[index, 2] <= cX or used[index, 1]+spacing+used[index,3] <= cY or cY+height+spacing<=used[index,1]: # check for no overlapping blobs
                    index = index + 1
                    tries = 0
                else:
                    cX = random.randint(margin, img_width-margin-width)
                    cY = random.randint(margin, img_height-margin-height)
                    index = 0
                    tries += 1
                    if tries > 5: # hangup, so restart adding blobs
                        #print("tries")
                        #print(tries)
                        img = np.zeros(img_height*img_width)
                        blob_idx = 0
                        used = np.zeros((num_blobs, 4))
 
            if blob_idx < num_blobs:
                used[index, 0] = cX
                used[index, 1] = cY
                used[index, 2] = width
                used[index, 3] = height

                for p in range(cY, cY+height):
                    for q in range(cX, cX+width):
                        img[p*img_width+q] = 255
                
            blob_idx += 1

        train[img_count] = img
        train_incr[img_count] = get_imgs_incr(used, img, N)

        img_count += 1
        i += 1

    np.set_printoptions(threshold=np.nan)
    
    return train, train_incr


def get_imgs_incr(used, img, N):
    """
    Get the images with N+1 blobs
    Arguments:
        img: image with N blobs
        used: positions of N blobs in img
    """
    img_incr_list = []
    for j in range(num_incr):
        img_incr = copy.deepcopy(img)
        index = 0
        cX = random.randint(margin, img_width-margin-width)
        cY = random.randint(margin, img_height-margin-height)

        while index < N:
            if cX+width+spacing <= used[index, 0] or used[index, 0]+spacing+used[index, 2] <= cX or used[index, 1]+spacing+used[index,3] <= cY or cY+height+spacing<=used[index,1]: # check for no overlapping blobs
                index = index + 1
            else:
                cX = random.randint(margin, img_width-margin-width)
                cY = random.randint(margin, img_height-margin-height)
                index = 0

        for p in range(cY, cY+height):
            for q in range(cX, cX+width):
                img_incr[p*img_width+q] = 255

        img_incr_list.append(img_incr)

    return img_incr_list

