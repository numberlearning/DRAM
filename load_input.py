import numpy as np
import matplotlib as mpl
import pickle
import create_data_natural
from model_settings import batch_size, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test
mpl.use('Agg')

import matplotlib.pyplot as plt


class InputData(object):
    """An object of InputData must have images and labels.

    Attributes:
        images: A list of lists of image pixels.
        labels: A list of one-hot label lists.
    """

    def __init__(self, folder="", images=[], labels=[]):
        """Return an InputData object."""
        self.folder = folder
        self.images = []
        self.labels = []
        self.areas = []
        self.length = 0

        self.labels_scalar = []
        self.labels_classification = []


    def get_train(self, even=None, min_blobs=1, max_blobs=1): # MT
        """Generate and get train images and labels."""
        self.images, self.labels, self.areas = create_data_natural.generate_data(even, min_blobs, max_blobs, scalar_output=True)
        self.length = len(self.images)


    def get_CTA(self, even=None, min_blobs=1, max_blobs=9):
        """Generate and get train images and labels."""
        self.images, self.labels, self.areas = create_data_natural.generate_data(even, min_blobs, max_blobs, CTA=True)
        self.length = len(self.images)
    
    
    def get_has_spacing(self, even=None, min_blobs=1, max_blobs=9):
        """Generate and get train images and labels."""
        self.images, self.labels, self.areas = create_data_natural.generate_data(even, min_blobs, max_blobs, has_spacing=True)
        self.length = len(self.images)


    def get_test(self, even=None, min_blobs=1, max_blobs=1): # MT
        """Generate and get test images and labels."""
        self.images, self.labels, self.areas = create_data_natural.generate_data(even, min_blobs, max_blobs, scalar_output=True) # MT
        self.length = len(self.images)


    def load_sample(self):
        """Load the sample set and labels."""
        self.load_images(self.folder + "/sampleSet.txt")
        self.load_labels(self.folder + "/sampleLabel.txt")


    def load_train(self):
        """Load the train set and labels."""
        self.load_images(self.folder + "/trainSet.txt")
        self.load_labels(self.folder + "/trainLabel.txt")


    def load_test(self):
        """Load the test set and labels."""
        self.load_images(self.folder + "/testSet.txt")
        self.load_labels(self.folder + "/testLabel.txt")


    def load_images(self, filename):
        """Load the image data"""
        self.images = self.load(filename)
        self.length = len(self.images)


    def load_labels(self, filename):
        """Load the image data"""
        self.labels = self.load(filename)


    def load_variable_position_only(self, filename):
        """Load the images with variable position only, and their scalar and classifier labels."""
        loaded = pickle.load( open( "variable_position_only_examples.p", "rb" ) )
        self.images, self.labels_scalar, self.labels_classifier = loaded

    def load_po(self, incremental=False):
        """Load po testset."""
        if incremental:
            loaded = pickle.load( open( "po_inc.p", "rb" ) )
        else:
            loaded = pickle.load( open( "po_ind.p", "rb" ) )
        self.images, self.labels_scalar, self.labels_classifier = loaded
        self.length = len(self.images)

    def load(self, filename):
        """Load the data from text file."""
        file = open(filename, "r")
        text = file.read()
        file.close()
        text = text.replace(']', '],').replace('],]', ']]').replace(']],', ']]')
        text = text.replace('.', ',').replace(',]', ']')
        aList = eval(text)
        return aList


    def get_length(self):
        """Return the number of images."""
        return self.length


    def next_batch(self, batch_size):
        """Returns a batch of size batch_size of data."""
        all_idx = np.arange(0, self.length)
        np.random.shuffle(all_idx)
        batch_idx = all_idx[:batch_size]
        batch_imgs = [self.images[i] for i in batch_idx]
        batch_lbls = [self.labels[i] for i in batch_idx]
        batch_areas = [self.areas[i] for i in batch_idx]
        return batch_imgs, batch_lbls, batch_areas

    def next_batch_po(self, batch_size):
        """Returns a batch of size batch_size of data."""
        all_idx = np.arange(0, self.length)
        np.random.shuffle(all_idx)
        batch_idx = all_idx[:batch_size]
        batch_imgs = [self.images[i] for i in batch_idx]
        batch_lbls_scalar = [self.labels_scalar[i] for i in batch_idx]
        batch_lbls_classifier = [self.labels_classifier[i] for i in batch_idx]
        return batch_imgs, batch_lbls_scalar, batch_lbls_classifier

    def next_batch_nds(self, batch_size):
        """Returns a batch of size batch_size of data."""
        all_idx = np.arange(0, self.length)
        batch_idx = all_idx[:batch_size]
        #print('actual number of images in batch: %d' % len(batch_idx))
        batch_imgs = [self.images[i] for i in batch_idx]
        batch_lbls = [self.labels[i] for i in batch_idx]
        batch_areas = [self.areas[i] for i in batch_idx]
        return batch_imgs, batch_lbls, batch_areas

    def split_data(self):
        """Returns the first half and latter half data of each number."""
        all_idx = np.arange(0, 9000)# self.length)
        nOfImgs = 1000
        fh_idx = all_idx[0:nOfImgs//2] # first half index
        lh_idx = all_idx[nOfImgs//2:nOfImgs] # latter half index
        for i in range(1,9):
            fh_idx = np.append(fh_idx, all_idx[nOfImgs*i:nOfImgs*i+nOfImgs//    2])
            lh_idx = np.append(lh_idx, all_idx[nOfImgs*i+nOfImgs//2:nOfImgs*    (i+1)])
        fh_imgs = [self.images[i] for i in fh_idx]
        fh_lbls = [self.labels[i] for i in fh_idx]
        lh_imgs = [self.images[i] for i in lh_idx]
        lh_lbls = [self.labels[i] for i in fh_idx]
        return fh_imgs, fh_lbls, lh_imgs, lh_lbls

    def print_img_at_idx(self, idx):
        """Prints the image at index idx."""
        img = self.images[idx]
        print_img(img)

    def get_label(self, idx):
        """Returns the label."""
        return self.labels[idx]

def test_this():
    """Test out this class."""
    myData = InputData()
    #myData.load_sample()
    myData.get_test(0, 1, 15)
    #x_train, y_train = myData.next_batch(10)
    #for i, img in enumerate(x_train):
    #    print_img(img)
    #    print(y_train[i])

    # are there images with greater numerosities?
    x_train, y_train, _ = myData.next_batch(100)
    for i, img in enumerate(x_train):
        if y_train[i] == 8:
            print_img(img)
            #print(y_train[i])


def save_data_variable_position_only():
    """Generate and save test images with variable position only, scalar labels, and classifier labels."""
    x, y_scalar, y_classifier = create_data_natural.generate_data_variable_position_only()
    variable_position_only_examples = (x, y_scalar, y_classifier)

    import pickle
    with open('variable_position_only_examples', 'wb') as fp:
        pickle.dump(variable_position_only_examples, fp)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_img(img):
    """Prints the image."""
    print('hello sharon')
    matrix = list(chunks(img, 100))
    plt.imshow(matrix, interpolation="nearest", origin="upper")
    #plt.colorbar()
    plt.show()

def main():
    #save_data_variable_position_only()
    #test_this()
    pass


main()
