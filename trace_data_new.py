import sys
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=np.nan)

"""
This program outputs 40x200 images with traces.
"""

def get_my_teacher():
    min_edge = 7
    max_edge = 7
    min_blobs = 1
    max_blobs = 5

        
    c = min_blobs
    total = 0
    while (c < max_blobs + 1):
       if c == 0:
           total = total + 1000
           c = c + 1
           continue
       cN = 1000#int(10/c)
       total = total + cN
       c = c + 1

    nOf_nBlob_imgs = 4
    img_height = 40   
    img_width = 200
    blob_spacing = 7
    edge_vertical_space = 10
    edge_horizontal_space = 10
    img_size = img_height * img_width   #row x columns
    n_labels = max_blobs  #features in a label
    all_traces = []
    train = np.zeros([total, img_size])   #training imgs
    label = np.zeros([total, n_labels])   #labels
    trace = []          #list(imgs) of set(traces for each img) of tuples(lx, ly)
    img_count = 0
    length = 0

    #create images for each number of blobs
    for n in range(min_blobs, max_blobs+1):
        num_blobs = n       #1 to 9
        num_imgs = 1000#int(10/n)      #number of images to generate for this number of blob
        
        #generate individual image
        for x in range(num_imgs):
            img = np.zeros(img_size)
            count = 0
            used = np.zeros([num_blobs, 4]) # x_coor, y_coor, width, height for each blob
            list_traces = []  #list of traces
            lxly_list = []   #x,y coordinate tuples of blobs in the image
	    
            #create info of each image
            while count < num_blobs:       
                height = random.randint(min_edge, max_edge)
                width = random.randint(min_edge, max_edge)
                if count is 0:
                    lx = random.randint(25, 35)
                else:
                    lx = random.randint(used[count-1, 0]+10, used[count-1, 0]+20)
                #lx = random.randint(edge_vertical_space, img_width - edge_horizontal_space - width)
                ly = random.randint(edge_vertical_space, img_height - edge_vertical_space - height)
                h = height
                w = width

                index = 0

                #find valid positions for the blobs
                """
                while index < count:
                    if lx+width+1 <= used[index, 0] or used[index, 0]+1+used[index,2] <= lx or used[index, 1]+1+used[index,3] <= ly or ly+height+1<=used[index,1]:
                        if index == count-1:
                            position = count - 1
                            #while position >= 0:
                            #    hori1 =  (lx >= used[position, 0] - width - 6) and (lx <= used[position, 0] - width - 1) 
                            #    hori2 =  (lx >= used[position, 0] + used[position,2] + 1) and (lx <= used[position, 0] + used[position,2] + 6) 
                            #    vert1 =  (ly >= used[position, 1] - height - 3) and (ly <= used[count-1, 1] - height - 1) 
                            #    vert2 =  (ly >= used[position, 1] + used[position, 3] + 1) and (ly <= used[position, 1] + used[position, 3] + 3)
                            #    if (hori1 or hori2) and (vert1 or vert2):    
                            #        index = index + 1
                            #        break
                            #    position = position - 1
                            #if index == count:
                            #    lx = random.randint(edge_horizontal_space, img_width - edge_horizontal_space - width)
                            #    ly = random.randint(edge_vertical_space, img_height - edge_vertical_space - height)
                            #    index = 0
                            while position >= 0:
                                distance = (lx - used[position, 0])**2 + (ly - used[position, 1])**2
                                if distance <= 338:
                                    index = index + 1
                                    break
                                position = position - 1
                            if index < count:
                                #lx = random.randint(
                                #lx = random.randint(edge_horizontal_space, img_width - edge_horizontal_space - width)
                                ly = random.randint(edge_vertical_space, img_height - edge_vertical_space - height)
                                index = 0
                        else:
                            index = index + 1
                    else:
			#left_bound1 = used[count, 0] - width - 5
	                #right_bound1 = used[
			#right_bound2 = used[count, 0] + used[count, 2] + 5
			#top_bound = used[count, 1] + used[count, 3] + 5
			#bottom_bound = used[count, 1] - height - 5
			#left = left_bound if left_bound > edge_horizontal_space else edge_horizontal_space 
                        lx = random.randint(edge_horizontal_space, img_width - edge_horizontal_space - width)
                        ly = random.randint(edge_vertical_space, img_height - edge_vertical_space - height)
                        index = 0
                    """

                used[count, 0] = lx
                used[count, 1] = ly 
                lxly_list.append((lx + int(width/2), ly + int(height/2)))
               # print(lxly_list)
                used[count, 2] = width
                used[count, 3] = height

                for p in range(ly, ly+height):
                    for q in range(lx, lx+width):
                        img[p*img_width+q] = 255

                count = count + 1
                #intensity = intensity + int(255/num_blobs)

            train[img_count] = img
            label[img_count, num_blobs-1] = 1
            img_count += 1
            img = img.reshape(img_height, img_width)
            #plt.imshow(img, interpolation="nearest", origin="upper")
            #plt.colorbar()
            #plt.title(label[img_count - 1])
            #plt.show()

            #create traces for each image
            for t in range(num_blobs):
                #trace_current = []
                l1 = lxly_list
                #if t % 2 == 0:
                #    l1.sort(key=lambda tup:tup[0])
                #    l1.sort(key=lambda tup:tup[1])
                #if t % 2 == 1:
                #    l1.sort(key=lambda tup:tup[0], reverse=True)
                #if t % 4 == 2:
                #    l1.sort(key=lambda tup:tup[0])
                #    l1.sort(key=lambda tup:tup[1], reverse=True)
                #if t % 4 == 3:
                #    l1.sort(key=lambda tup:tup[0], reverse=True)
                #    l1.sort(key=lambda tup:tup[1], reverse=True)
                #for index in range(num_blobs):
                    #trace_current.append(l1[index])
                #    minimum = 10000000000
                #    for p in range(1,num_blobs-index):   
                #        diff = (l1[index+p][0]-l1[index][0])**2 + (l1[index+p][1]-l1[index][1])**2
                #        if diff < minimum:
                #            minimum = diff
                #            temp = l1[index+1]
                #            l1[index+1] = l1[index+p]
                #            l1[index+p] = temp
                #        p = p + 1
                #    index = index+1
                v =int(255 / num_blobs)
                im = np.zeros((img_height, img_width))
                for j in range(num_blobs):
                    for y in range(-2, 3):
                        for x in range(-2, 3):    
                            im[l1[j][1]+x][l1[j][0]+y] = v
                    v = v + int(255 / num_blobs)
                trace_vector = np.zeros(img_size)
                v = int(255/num_blobs)
                for j in range(num_blobs):
                    for y in range(-2, 3):
                        for x in range(-2, 3):
                            trace_vector[(l1[j][1]+y)*img_width+(l1[j][0]+x)]=v
                            #print((l1[j][0]+y)*img_width + (l1[j][1]+x))
                    v = v + int(255/num_blobs)
                imggg = trace_vector.reshape(img_height, img_width)
                #print(im)
            #    plt.imshow(im, interpolation="nearest", origin="upper")
            #    plt.colorbar()
            #    plt.title(label[img_count - 1])
            #    plt.show()
                list_traces.append(l1)
                length = length + 1
                all_traces.append(trace_vector)
            
            trace.append(list_traces)
            #print(trace)

    return all_traces, train, label, trace, length
