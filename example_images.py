from DRAMcluttered import convertTranslated
from tensorflow.examples.tutorials import mnist
import matplotlib.pyplot as plt

data = mnist.input_data.read_data_sets("mnist", one_hot=True).train
for i in range(500):
	x, y = data.next_batch(1)

img = convertTranslated(x)[0].reshape((100, 100))

plt.imshow(img)

plt.show()