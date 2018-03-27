#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from model_settings import img_height, img_width, min_edge, max_edge

dims = [img_height, img_width]
img_size = dims[1]*dims[0] # canvas size
read_n = 13  # N x N attention window
read_size = read_n*read_n
# delta, sigma2
delta_1 = 10#max(dims[0],dims[1])*1.5/(read_n-1) 
sigma2_1 = delta_1*delta_1/4 # sigma=delta/2 
delta_2 = 3#max(dims[0],dims[1])/2/(read_n-1)
sigma2_2=delta_2*delta_2/4 # sigma=delta/2
# normfac
normfac_1 = 1.0/np.sqrt(2*np.pi*sigma2_1)
normfac_2 = 1.0/np.sqrt(2*np.pi*sigma2_2)
# attention window center
gx = dims[0]/2
gy = dims[1]/2
# mu: filter location
grid_i = np.arange(read_n)
mu_x_1 = np.reshape(gx + (grid_i - read_n / 2 + 0.5) * delta_1, (-1,1))
mu_y_1 = np.reshape(gy + (grid_i - read_n / 2 + 0.5) * delta_1, (-1,1)) 
mu_x_2 = np.reshape(gx + (grid_i - read_n / 2 + 0.5) * delta_2, (-1,1))
mu_y_2 = np.reshape(gy + (grid_i - read_n / 2 + 0.5) * delta_2, (-1,1))
# (a,b): a point in the input image
a = np.reshape(np.arange(dims[0]), (1, -1)) 
b = np.reshape(np.arange(dims[1]), (1, -1))

input_img = np.ones(img_height*img_width) # input img size
blob_size = max_edge
width = blob_size
height = blob_size
cX = img_width/2 - blob_size/2
cY = img_height/2 - blob_size/2

for p in range(int(cY), int(cY+height)):
    for q in range(int(cX), int(cX+width)):
        input_img[p*img_width+q] = 255

input_img = np.reshape(input_img,(dims[0],dims[1]))

Fx_1 = normfac_1 * np.exp(-np.square(a - mu_x_1) / (2*sigma2_1))
Fy_1 = normfac_1 * np.exp(-np.square(b - mu_y_1) / (2*sigma2_1))
Fx_2 = normfac_2 * np.exp(-np.square(a - mu_x_2) / (2*sigma2_2)) 
Fy_2 = normfac_2 * np.exp(-np.square(b - mu_y_2) / (2*sigma2_2)) 

# filter_img
Fxt_1=np.transpose(Fx_1)
filter_img_1=np.matmul(Fy_1, np.matmul(input_img, Fxt_1))

Fxt_2=np.transpose(Fx_2)
filter_img_2=np.matmul(Fy_2, np.matmul(input_img, Fxt_2))

actfac_1 = filter_img_1[read_n//2,read_n//2]
actfac_2 = filter_img_2[read_n//2,read_n//2]

#print("filter_img_1:", filter_img_1)
print("actfac_1:", actfac_1)
#print("filter_img_2:", filter_img_2)
print("actfac_2:", actfac_2)

