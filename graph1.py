import analysis
import matplotlib.pyplot as plt
import numpy as np

glimpses = 10
output_size = 9
a = np.zeros([glimpses, output_size, output_size + 1])

print("graph1.py")
print("All-STEP", analysis.accuracy_stats(40, True))
print("LAST-STEP", analysis.accuracy_stats(40, False))

a = analysis.accuracy_stats(40, False)

plt.figure()
#b = [200, 201, 202, 203, 204, 205, 206, 207, 208]
for i in range(output_size):
    #k = b[i]
    #plt.subplot(k)
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'bo', a[glimpses-1][i], 'k')
    plt.show()
