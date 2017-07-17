import numpy as np
import matplotlib as mpl
import create_data
import load_input

mpl.use('Agg')
import matplotlib.pyplot as plt


class Teacher(object):
    """An object of Teacher must have images and labels.

    Attributes:
        images: A list of lists of image pixels.
        labels: A list of lists of image pixels after a timestep is completed.
    """

    def __init__(self, folder="", images=[], labels=[]):
        """Return a Teacher object."""

        self.folder = folder
        self.images = []
        self.explode_images = []
        self.labels = []
        self.explode_labels = []
        self.length = 0
        self.explode_length = 0


    def get_train(self, even=None):
        """Generate and get train images and labels."""

        self.images, self.labels = create_data.generate_data(even)
        self.length = len(self.images)
        self.explode()


    def get_test(self, even=None):
        """Generate and get train images and labels."""

        self.get_train(even)


    def load_sample(self):
        """Load the sample set and labels."""

        self.load_images(self.folder + "/sampleSet.txt")
        self.load_labels(self.folder + "/sampleLabel.txt")


    def load_images(self, filename):
        """Load the image data"""

        self.images = self.load(filename)
        self.length = len(self.images)
        self.explode()


    def load_labels(self, filename):
        """Load the image data"""

        self.labels = self.load(filename)


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
        return batch_imgs, batch_lbls


    def print_img_at_idx(self, idx):
        """Prints the image at index idx."""

        img = self.images[idx]
        print_img(img)


    def get_label(self, idx):
        """Returns the label."""

        return self.labels[idx]


    def explode(self):
        """Create the chain of explode_images and explode_labels formed by counting across two timestamps."""

        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

        for i, image in enumerate(self.images):
            for j, chain in enumerate(list(self.labels[i])):
                for k, link in enumerate(chain):

                    if k is 0:
                        explode_image = image[:]
                        explode_image = explode_image.append("zero", 0, 0)
                    
                    if k is not len(chain) - 1:
                        next_image = explode_label = explode_image[:-3] # remove word, x_coor, and y_coor

                        word = words[k]
                        x_coor = link[0]
                        y_coor = link[1]

                        explode_label.append(word, x_coor, y_coor)

                        # Add the current and the next frames to the data
                        self.explode_images.append(explode_image)
                        self.explode_labels.append(explode_label)
                        explode_image = next_image
                    else:
                        # No action is made by the teacher after counting is done
                        explode_label = explode_image
                        self.explode_images.append(explode_image)
                        self.explode_labels.append(explode_label)

        self.explode_length = len(self.explode_image)


    def explode_chained(self):
        """Create the chain of explode_images and explode_labels formed by counting across all timestamps."""

        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

        for i, image in enumerate(self.images):
            for j, chain in enumerate(list(self.labels[i])):

                # initialize new lists to keep track of another chain
                explode_image_chain = list()
                explode_label_chain = list()

                for k, link in enumerate(chain):

                    if k is 0:
                        explode_image = image[:]
                        explode_image = explode_image.append("zero", 0, 0)
                    
                    if k is not len(chain) - 1:
                        next_image = explode_label = explode_image[:-3]

                        word = words[k]
                        x_coor = link[0]
                        y_coor = link[1]

                        explode_label.append(word, x_coor, y_coor)

                        # Label the image with the action of the teacher given the image as input
                        explode_image_chain.append(explode_image)
                        explode_label_chain.append(explode_label)
                        explode_image = next_image
                    else:
                        explode_label = explode_image
                        explode_image_chain.append(explode_image)
                        explode_label_chain.append(explode_label)

                # Add entire counting sequence grouped together from start to finish
                self.explode_labels.append(explode_label_chain)
                self.explode_images.append(explode_image_chain)

        self.explode_length = len(self.explode_image)



def test_this():
    """Test out this class."""
    myData = Teacher()
    myData.load_sample()
    print(myData.get_length())
    x_train, y_train = myData.next_batch(10)
    for i, img in enumerate(x_train):
        print_img(img)
        print(y_train[i])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_img(img):
    """Prints the image."""
    matrix = list(chunks(img, 100))
    plt.imshow(matrix, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()

