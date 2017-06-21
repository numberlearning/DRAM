<<<<<<< HEAD
import analysis
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

########################################################
#MATPLOTLIB
#################
x = a[:, 0, :]
plt.hist(x, normed=True, bins=output_size)
plt.ylabel('Prediction Distribution for Input of 1 Blob')
plt.show()
