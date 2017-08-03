import numpy as np
import matplotlib as mpl
import trace_data_new

#mpl.use("Agg")
#import matplotlib.pyplot as plt


class TraceData(object):
    """An object of InputData must have images and labels.

    Attributes:
        images: A list of lists of image pixels.
        labels: A list of one-hot label lists.
    """

    def __init__(self, folder=".", images=[], labels=[]):
        """Return an InputData object."""
        self.folder = folder
        self.images = []
        self.length = 0


    def get_train(self, even=None):
        """Generate and get train images and labels."""
        self.images, _, _, _, _ = trace_data_new.get_my_teacher()
        self.length = len(self.images)


    def get_test(self, even=None):
        """Generate and get train images and labels."""
        self.get_train(even)


    def get_length(self):
        """Return the number of images."""
        return self.length


    def next_batch(self, batch_size):
        """Returns a batch of size batch_size of data."""
        all_idx = np.arange(0, self.length)
        np.random.shuffle(all_idx)
        batch_idx = all_idx[:batch_size]
        batch_imgs = [self.images[i] for i in batch_idx]
        return batch_imgs

    def print_img_at_idx(self, idx):
        """Prints the image at index idx."""
        img = self.images[idx]
        print_img(img)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


