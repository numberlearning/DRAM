import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import analysis

glimpses = 3
output_size = 4
a = np.zeros([glimpses, output_size, output_size + 1])
b = np.asarray([1, 2, 3, 4])
plt.hist(b)
plt.show()

print("graph1.py")
a = analysis.accuracy_stats(120000, False)

print(a)
plt.figure()
#b = [200, 201, 202, 203, 204, 205, 206, 207, 208]
for i in range(output_size):
    #k = b[i]
    #plt.subplot(k)
    plt.plot([pred_num for pred_num in range(output_size + 1)], a[glimpses-1][i])
    plt.ylabel('Count')
    plt.xlabel('Prediction')
    plt.title('Prediction Distribution at ' + str(i + 1) + ' Blobs')
    plt.show()
