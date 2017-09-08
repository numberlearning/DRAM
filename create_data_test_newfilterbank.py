import sys
import numpy as np
import random

num_img = 100
read_n = 15

# generate blobs
num_blobs = 1
blob_size = 5
# top left corner coordinate
cX = 60
cY = 60
height = blob_size
width = height # square blobs
img = np.zeros(10000)
for i in range(cX, cX+width):
    for j in range(cY, cY+height):
        img[i*100+j] = 255
    
# generate attention window center
gx = np.zeros(num_img)
gy = np.zeros(num_img)
for k in range(num_img):
    gx[k] = random.randint(1,99)
    gy[k] = random.randint(1,99)

origin_img = img
filter_img = np.ones(10000)

