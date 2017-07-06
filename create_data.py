import sys
import numpy as np
import random
from model_settings import min_edge, max_edge, min_blobs, max_blobs
n_labels = max_blobs - min_blobs + 1

c = min_blobs
total = 0
while (c < max_blobs + 1):
   if c == 0:
       total = total + 1000
       c = c + 1
       continue
   cN = int(10000/c)
   total = total + cN
   c = c + 1


def set_size(num_blobs, even=None):
    if even is None:
        if num_blobs == 0:
            return 1000
        else:
            return int(10000/num_blobs)
    else:
        return 1000
 
def generate_data(even=None):      
    num_blobs = min_blobs
    i = 0
    if even is None:
        train = np.zeros([total, 10000])
        label = np.zeros([total, n_labels])
    else:
        train = np.zeros([1000 * n_labels, 10000])
        label = np.zeros([1000 * n_labels, n_labels])
        
    while (num_blobs < max_blobs + 1):

        nOfItem = set_size(num_blobs, even)

        if num_blobs == 0:
           for n in range(nOfItem):
               train[i] = np.zeros(10000)
               label[i, num_blobs] = 1
               i = i +1
           num_blobs = num_blobs + 1
           continue

        for n in range(nOfItem):
            a = np.zeros(10000)
            count = 0
            used = np.zeros((num_blobs, 4))
    
            while count < num_blobs: 
                height = random.randint(min_edge, max_edge)
                width = random.randint(min_edge, max_edge)
                cX = random.randint(1, 99-width)
                cY = random.randint(1, 99-height)
                h = height 
                w = width
                
                index = 0
                
                while index < count:
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

                for p in range(cX, cX+width):
                    for q in range(cY, cY+height): 
                        a[p*100+q] = 255
                count = count + 1
          
            train[i] = a
            if min_blobs == 0:
                label[i, num_blobs] = 1
            elif min_blobs == 1:
                label[i, num_blobs - 1] = 1
            else:
                print("Hey, you! Min blobs should be one or zero!!!!")
            i = i + 1
          
        num_blobs = num_blobs + 1
    
    np.set_printoptions(threshold=np.nan)

#    if empty_img:
#        train = np.vstack((train, np.zeros((1000, 10000))))
#        for empty_idx in range(1000):
#            empty_label = np.zeros(max_blobs + 1)
#            empty_label[0] = 1
#            label = np.vstack((label, empty_label))

    return train, label
