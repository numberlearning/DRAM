import analysis
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

print("graph.py")
glimpses = 3
output_size = 9
a = np.zeros([glimpses, output_size, output_size + 1])

a = analysis.accuracy_stats(40000, False)
print(a)

########################################################
#MATPLOTLIB
#################
x = a[glimpses - 1, 0, :]
plt.hist(x, normed=True, bins=output_size)
plt.ylabel('Prediction Distribution for Input of 1 Blob')
plt.show()
