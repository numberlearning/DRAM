import sys
import numpy as np
import random
from model_settings import img_height, img_width, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

def get_S(max_blobs):
    """Get the denominator of proportion of blobs with a specific number of blobs."""

    S = 0.0
    for i in range(1, max_blobs):
        S += i ** (-2)
    return S


def get_size(testing, num_blobs, max_blobs):
    """Get amount of images for each number"""
    
    if testing:
        return 1000 # even: make 1000 images for each number
    else:
        # testing models' performance on training data 
        return 1000#int(10000*(num_blobs**(-2))/get_S(max_blobs)) # uneven distribution


def get_total(testing, min_blobs, max_blobs): # MT
    """Get total number of images."""

    c = min_blobs
    total = 0
    while (c < max_blobs + 1):
       total = total + get_size(testing, c, max_blobs)       
       c = c + 1
    return total


def get_k(testing=False):
    """Get the constant scale factor for an individual blob.""" 
    
    if testing:
        return 3 # testing: k is set to 3 throughout
    else:
        return random.randint(2, 4) # training: k is chosen from the uniform distribution of discrete values 2, 3, and 4


def get_s(testing=False, n=None):
    """Get the relative scale factor."""

    if testing:
        if n is not None:
            return random.uniform(.8*3 * (n ** (-.5)), 1.2*3 * (n ** (-.5))) # first 500 displays
        else:
            return random.uniform(.8, 1.2) # latter 500 displays
    else:
        return random.uniform(0.5*2 * (n ** (-.3154)), 1.5*2 * (n ** (-.3154))) # new DAA data

def pir(x):
    """Perform probabilistic integer rounding"""

    f, rem = divmod(x, 1)
    retval = int(f)
    
    if np.random.uniform() < rem:
        retval = retval + 1

    return retval


def get_dims(testing, i, num_blobs):
    """Get the dimensions of the ith blob."""

    k1 = get_k(testing)
    if testing:
        k2 = k1
    else:
        k2 = get_k(testing)

    if testing and i >= 500:
        s = get_s(testing)
    elif testing and i < 500: # equal total area
        s = get_s(testing, num_blobs)
    elif not testing:
        s = get_s(False, num_blobs)

    width = pir(k1 * s)
    height = pir(k2 * s)

    return width, height

def generate_data(incremental=False):
    """
    This function generates 1000 examples for each numerosity.
    Only the positions of the blobs is varied.
    The size of each blob is 3x3.
    Returns the images, scalar labels and classifier labels occupied by all the blobs of an image.
    """
    testing = True
    margin = 5
    spacing = 20
    width = height = 3
    min_blobs = 1
    max_blobs = 9
    num_labels = max_blobs - min_blobs + 1

    if incremental:
        return get_po_inc(margin, spacing, width, height, min_blobs, max_blobs)
    else:
        return get_po_ind(margin, spacing, width, height, min_blobs, max_blobs)


def get_blob(blob_cnt, margin, spacing, width, height, used):
    """
    Get positions x, y of top left corner of new blob,
    given an image with already blob_cnt blobs,
    placed margin from the edge of the image,
    placed spacing apart from other blobs,
    of dimensions width x height.
    """
    # Get random coordinates for the top-left corner of a blob
    x = random.randint(margin, img_width-margin-width)
    y = random.randint(margin, img_height-margin-height)

    # Check that new blob doesn't overlap with any of the other blobs
    blob_idx = 0
    tries = 0
    while blob_idx < blob_cnt:
        if x+width+spacing <= used[blob_idx, 0] or used[blob_idx, 0]+spacing+used[blob_idx, 2] <= x or used[blob_idx, 1]+spacing+used[blob_idx,3] <= y or y+height+spacing<=used[blob_idx,1]:
            blob_idx = blob_idx + 1
            tries = 0
        else:
            # Overlapped with some blob, regenerate coordinates for new blob
            x = random.randint(margin, img_width-margin-width)
            y = random.randint(margin, img_height-margin-height)
            blob_idx = 0
            tries += 1

            # Hangup, so restart adding blobs
            if tries > 5:
                raise Exception("Hangup, blob cannot be added")
            
    return x, y


def get_img(N, margin, spacing, width, height):
    """
    Create an image with N blobs of width x height,
    a margin along the edge of the image,
    and spacing between the blobs.
    """
    img = np.zeros(img_height*img_width) # initialize empty image
    blob_cnt = 0 # count of blobs in each image 
    used = np.zeros((N, 4)) # bookkeep positions of blobs
    while blob_cnt < N:
        try:
            x, y = get_blob(blob_cnt, margin, spacing, width, height, used) 
        except:
            img = np.zeros(img_height*img_width)
            blob_cnt = 0
            used = np.zeros((N, 4))

        # Bookkeep new blob position information
        used[blob_cnt, 0] = int(x)
        used[blob_cnt, 1] = int(y)
        used[blob_cnt, 2] = width
        used[blob_cnt, 3] = height

        # Fill in pixels for new blob
        for p in range(y, y+height):
            for q in range(x, x+width):
                img[p*img_width+q] = 255
        blob_cnt += 1
    return img, used.astype(int)


def get_po_ind(margin, spacing, width, height, min_blobs, max_blobs):
    """
    Get testing dataset where the only variable is position,
    and images of each numerosity are generated independently.
    """
    total = get_total(True, min_blobs, max_blobs)
    images = np.zeros([total, img_height*img_width])
    label_scalar = np.zeros([total, 1])
    label_classifier = np.zeros([total, max_blobs-min_blobs+1])

    curr_num = min_blobs # current numerosity

    total_img_cnt = 0
    while curr_num < max_blobs + 1:
        num_img_cnt = 0
        while num_img_cnt < 1000:
            img, _ = get_img(curr_num, margin, spacing, width, height)
            images[total_img_cnt] = img
            label_scalar[total_img_cnt] = curr_num
            label_classifier[total_img_cnt, curr_num-1] = 1
            num_img_cnt += 1
            total_img_cnt += 1
            #if total_img_cnt % 1000 == 0:
            #    print("total_img_cnt: %d" % total_img_cnt)

        curr_num += 1

    np.set_printoptions(threshold=np.nan)
    return images, label_scalar, label_classifier


def get_po_inc(margin, spacing, width, height, min_blobs, max_blobs):
    """
    Get testing dataset where the only variable is position,
    and 1000 sets of images where blobs are placed incrementally.
    """
    total = get_total(True, min_blobs, max_blobs)
    num_sets = 1000
    images = np.zeros([total, img_height*img_width])
    label_scalar = np.zeros([total, 1])
    label_classifier = np.zeros([total, max_blobs-min_blobs+1])

    set_cnt = 0
    total_img_cnt = 0
    while set_cnt < num_sets:
        for curr_num in range(max_blobs, 0, -1):
            if curr_num is max_blobs:
                img, used = get_img(max_blobs, margin, spacing, width, height)
            else:
                x, y, _, _ = used[curr_num]
                for p in range(y, y+height):
                    for q in range(x, x+width):
                        img[p*img_width+q] = 0
            images[total_img_cnt] = img[:]
            label_scalar[total_img_cnt] = curr_num
            label_classifier[total_img_cnt, curr_num-1] = 1
            total_img_cnt += 1

        set_cnt += 1

    np.set_printoptions(threshold=np.nan)
    return images, label_scalar, label_classifier
