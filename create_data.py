import sys
import numpy as np
import random
from model_settings import min_edge, max_edge, max_blobs

c = 1
total = 0
while (c < 10):
   cN = int(10000/c)
   total = total + cN
   c = c + 1

def set_size(num_blobs, even=None):
    if even is None:
        return int(10000/num_blobs)
    else:
        return 1000
 
def generate_data(even=None):        
    num_blobs = 1
    i = 0
    if even is None:
        train = np.empty([total, 10000])
        label = np.empty([total, max_blobs])
    else:
        train = np.empty([9000, 10000])
        label = np.empty([9000, max_blobs])
        
    while (num_blobs < max_blobs + 1):

        nOfItem = set_size(num_blobs, even)

        for n in range(1, nOfItem + 1):
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
            label[i, num_blobs-1] = 1
            i = i + 1
          
        num_blobs = num_blobs + 1
    
    np.set_printoptions(threshold=np.nan)
    return train, label
