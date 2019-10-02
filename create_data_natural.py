import sys
import numpy as np
import random
from model_settings import img_height, img_width, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

CTA_dims = [(9, 9), (7, 6), (5, 5), (5, 4), (4, 4), (4, 3), (4, 3), (3, 3), (3, 3), (3, 3), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2)]

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
        #return random.uniform(.5, 1.5) # new CAA data
        return random.uniform(0.5*2 * (n ** (-.3154)), 1.5*2 * (n ** (-.3154))) # new DAA data
        #return random.uniform(.8*2 * (n ** (-.3154)), 1.2*2 * (n ** (-.3154))) # previous DAA data

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

def generate_data(testing, min_blobs, max_blobs, density=False, CTA=False, has_spacing=False): # MT
    n_labels = max_blobs_train - min_blobs_train + 1
    total = get_total(testing, min_blobs, max_blobs)
    train = np.zeros([total, img_height*img_width]) # input img size
    label = np.zeros([total, n_labels])
    #total_area_blobs = np.zeros([total, 1])
    #mean_area_blobs = np.zeros([total, 1])
    num_blobs = min_blobs
    img_count = 0
    
    #average_edge = []

    while (num_blobs < max_blobs + 1):

        nOfItem = get_size(testing, num_blobs, max_blobs) # testing: 1000; training, 10000*num_blobs**(-2)/sum(i**(-2))
        i = 0 # amount of images for each blob number
        #sum_edge = 0.0
        #count_edge = 0.0

        while (i < nOfItem):
            img = np.zeros(img_height*img_width) # input img size
            num_count = 0 # amount of blobs in each image 
            used = np.zeros((num_blobs, 4)) # check overlapping
            #total_area = 0.0
            d_edge = 3
            while num_count < num_blobs:
                if CTA:
                    width, height = CTA_dims[num_blobs-1]
                elif density:
                    width = d_edge
                    height = d_edge
                elif has_spacing:
                    width = d_edge
                    height = d_edge
                else:
                    width, height = get_dims(testing, i, num_blobs)

                margin = 10
                if has_spacing:
                    if num_blobs < 10:
                        spacing = 20
                    else:
                        spacing = 1
                else:
                    spacing = 1

                if density:
                    cX = random.randint(int(50-(d_edge/2+1.5)*num_blobs), int(50+(d_edge/2+1.5)*num_blobs))
                    cY = random.randint(int(50-(d_edge/2+1.5)*num_blobs), int(50+(d_edge/2+1.5)*num_blobs))
                else:
                    cX = random.randint(margin, img_width-margin-width) # top left corner
                    cY = random.randint(margin, img_height-margin-height)

                index = 0
                tries = 0

                while index < num_count:
                    if cX+width+spacing <= used[index, 0] or used[index, 0]+spacing+used[index, 2] <= cX or used[index, 1]+spacing+used[index,3] <= cY or cY+height+spacing<=used[index,1]: # check for no overlapping blobs
                        index = index + 1
                        tries = 0
                    else:
                        if density:
                            cX = random.randint(int(50-(d_edge/2+1.5)*num_blobs), int(50+(d_edge/2+1.5)*num_blobs))
                            cY = random.randint(int(50-(d_edge/2+1.5)*num_blobs), int(50+(d_edge/2+1.5)*num_blobs))
                        else:
                            cX = random.randint(margin, img_width-margin-width)
                            cY = random.randint(margin, img_height-margin-height)
                        index = 0
                        tries += 1
                        if tries > 5: # hangup, so restart adding blobs
                            #print("tries")
                            #print(tries)
                            img = np.zeros(img_height*img_width)
                            num_count = 0
                            used = np.zeros((num_blobs, 4))
                    


                used[index, 0] = cX
                used[index, 1] = cY
                used[index, 2] = width
                used[index, 3] = height
                #total_area += width * height

                #sum_edge += width
                #count_edge += 1.0
                #sum_edge += height
                #count_edge += 1.0
                for p in range(cY, cY+height):
                    for q in range(cX, cX+width):
                        img[p*img_width+q] = 255
                num_count += 1

            train[img_count] = img
            label[img_count, num_blobs - 1] = 1
            #total_area_blobs[img_count] = total_area
            #mean_area_blobs[img_count] = total_area / num_blobs
            img_count += 1
            #if img_count % 1000 == 0:
            #    print("img_count: %d" % img_count)
            i += 1

        #average_edge.append(sum_edge / count_edge)
        num_blobs += 1

    np.set_printoptions(threshold=np.nan)
    
    return train, label#, total_area_blobs, mean_area_blobs, average_edge
