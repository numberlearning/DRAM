import sys
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_my_teacher():
    min_edge = 1
    max_edge = 1
    min_blobs = 1
    max_blobs = 9

        
    c = min_blobs
    total = 0
    while (c < max_blobs + 1):
       if c == 0:
           total = total + 1000
           c = c + 1
           continue
       cN = int(10/c)
       total = total + cN
       c = c + 1

    nOf_nBlob_imgs = 4
    img_edge = 10
    img_size = img_edge * img_edge
    n_labels = max_blobs
    train = np.zeros([total, img_size])   #training imgs
    label = np.zeros([total, n_labels])   #labels
    trace = []          #list(imgs) of set(traces for each img) of tuples(lx, ly)
    img_count = 0

    #create 4 images
    for n in range(min_blobs, max_blobs+1):
        num_blobs = n
        num_imgs = int(10/n)
        for x in range(num_imgs):
            img = np.zeros(img_size)
            count = 0
            used = np.zeros([num_blobs, 4]) # x_coor, y_coor, width, height for each blob
            list_traces = []  #set of traces
            lxly_list = []
            #create info of each image
            while count < num_blobs:       
                height = random.randint(min_edge, max_edge)
                width = random.randint(min_edge, max_edge)
                lx = random.randint(1, 9-width)
                ly = random.randint(1, 9-width)
                h = height
                w = width

                index = 0

                while index < count:
                    if lx+width+1 <= used[index, 0] or used[index, 0]+1+used[index,2] <= lx or used[index, 1]+1+used[index,3] <= ly or ly+height+1<=used[index,1]:
                        index = index + 1
                    else:
                        lx = random.randint(1, 9-width)
                        ly = random.randint(1, 9-height)
                        index = 0
                
                used[index, 0] = lx
                used[index, 1] = ly
                lxly_list.append((lx, ly))
               # print(lxly_list)
                used[index, 2] = width
                used[index, 3] = height

                for p in range(lx, lx+width):
                    for q in range(ly, ly+height):
                        img[p*10+q] = 255

                count = count + 1

            train[img_count] = img
            label[img_count, num_blobs-1] = 1
            img_count += 1

            for t in range(num_blobs):
                #trace_current = []
                l1 = lxly_list
                if t % 4 == 0:
                    l1.sort(key=lambda tup:tup[0])
                    l1.sort(key=lambda tup:tup[1])
                if t % 4 == 1:
                    l1.sort(key=lambda tup:tup[0], reverse=True)
                    l1.sort(key=lambda tup:tup[1])
                if t % 4 == 2:
                    l1.sort(key=lambda tup:tup[0])
                    l1.sort(key=lambda tup:tup[1], reverse=True)
                if t % 4 == 3:
                    l1.sort(key=lambda tup:tup[0], reverse=True)
                    l1.sort(key=lambda tup:tup[1], reverse=True)
                for index in range(num_blobs):
                    #trace_current.append(l1[index])
                    minimum = 10000
                    for p in range(1,num_blobs-index):   
                        diff = (l1[index+p][0]-l1[index][0])**2 + (l1[index+p][1]-l1[index][1])**2
                        if diff < minimum:
                            minimum = diff
                            temp = l1[index+1]
                            l1[index+1] = l1[index+p]
                            l1[index+p] = temp
                        p = p + 1
                    index = index+1
                v =int(255 / num_blobs)
                im = np.zeros((10, 10))
                for j in range(num_blobs):
                    im[l1[j][0]][l1[j][1]] = v
                    v = int(v + 255 / num_blobs)
                #print(im)
                #plt.imshow(im, interpolation="nearest", origin="upper")
                #plt.colorbar()
                #plt.title(label[img_count - 1])
                #plt.show()
                list_traces.append(l1)

            
            trace.append(list_traces)
            #print(trace)

    return train, label, trace

    # np.set_printoptions(threshold=np.nan)
    # sys.stdout = open("tTrain.txt", "w")
    # print(train)
    # sys.stdout.close()
    # sys.stdout = open("tLabel.txt", "w")
    # print(label)
    # sys.stdout.close()
    # sys.studout = open("tTrace.txt", "w")


