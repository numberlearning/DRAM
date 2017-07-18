import sys
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import load_teacher

def get_my_teacher():
    min_edge = 1
    max_edge = 1
    min_blobs = 1
    max_blobs = 4

    c = 4
    nOf_nBlob_imgs = 4
    img_edge = 10
    img_size = img_edge * img_edge
    n_labels = max_blobs
    train = np.zeros([nOf_nBlob_imgs, img_size])   #training imgs
    label = np.zeros([nOf_nBlob_imgs, n_labels])   #labels
    trace = []          #list(imgs) of list(traces for each img) of tuples(lx, ly)

    #create 4 images
    for n in range(4):
        num_blobs = max_blobs
        img = np.zeros(img_size)
        count = 0
        used = np.zeros([num_blobs, 4])
        list_traces = []  #list of traces
        lxly_list = []
        #create info of each image
        while count < max_blobs:       
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
            #print(lxly_list)
            used[index, 2] = width
            used[index, 3] = height

            for p in range(lx, lx+width):
                for q in range(ly, ly+height):
                    img[p*10+q] = 255

            count = count + 1

        train[n] = img
        label[n, max_blobs-1] = 1
        
        #find traces from all possible starting point

        for t in range(4):
            trace1 = [] #list of tuples of t
            l1 = []
            #print(lxly_list)
            #print(lxly_list.sort(key=lambda tup:tup[0]))
            if t % 4 == 1:
                l1 = lxly_list
                l1.sort(key=lambda tup:tup[0])
                #print(lxly_list.sort(key=lambda tup:tup[0]))
                #print(l1)
            if t % 4 == 2:
                l1 = lxly_list
                l1.sort(key=lambda tup:tup[0], reverse=True)
                #print(lxly_list.sort(key=lambda tup:tup[0], reverse=True))
                #print(l1)
            if t % 4 == 3:
                l1 = lxly_list
                l1.sort(key=lambda tup:tup[1])
                #print(lxly_list.sort(key=lambda tup:tup[1]))
                #print(l1)
            if t % 4 == 0:
                l1 = lxly_list
                l1.sort(key=lambda tup:tup[1], reverse=True)
                #print(lxly_list.sort(key=lambda tup:tup[1], reverse=True))
                #print(l1)
            l1trace = []
#            print(l1)
            im = np.zeros((10, 10))
            for i in range(4):
                diff = (l1[i][0] - l1[0][0])*(l1[i][0]-l1[0][0]) + (l1[i][1] - l1[0][1])*(l1[i][1]- l1[0][1])
                l1trace.append((l1[i], diff))
            l1trace.sort(key=lambda tup:tup[1])
#            print(l1trace)
            v = 50
            for j in range(4):
                trace1.append(l1trace[j][0])
                im[l1trace[j][0][0]][l1trace[j][0][1]] = v
                v = v + 50

            # Print the image visually
            #print(im)
            #plt.imshow(im, interpolation="nearest", origin="upper")
            #plt.colorbar()
            #plt.show()

            list_traces.append(trace1)
            #print(list_traces)
            
        trace.append(list_traces)


#    np.set_printoptions(threshold=np.nan)
#    sys.stdout = open("tTrain.txt", "w")
#    print(train)
#    sys.stdout.close()
#    sys.stdout = open("tLabel.txt", "w")
#    print(label)
#    sys.stdout.close()
#    sys.studout = open("tTrace.txt", "w")
    return train, label, trace


