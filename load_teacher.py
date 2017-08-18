import numpy as np
import matplotlib as mpl
#import trace_data
import trace_data_new as trace_data

mpl.use('Agg')
import matplotlib.pyplot as plt

z_size = 5


class Teacher(object):
    """An object of Teacher must have images and traces.

    Attributes:
        images: A list of lists of image pixels.
        labels: A list of one-hot label lists.
        traces: A list of lists of image pixels after a timestep is completed.
    """

    def __init__(self, folder="", images=[], labels=[], traces=[]):
        """Return a Teacher object."""

        self.folder = folder
        self.images = []
        self.explode_images = []
        self.labels = []
        self.explode_lbls = []
        self.explode_labels = []
        self.traces = []
        self.explode_traces = []
        self.explode_counts = []
        self.length = 0
        self.explode_length = 0


    def get_train(self, even=None):
        """Generate and get train images and traces."""

        #self.images, self.labels, self.traces = trace_data.get_my_teacher()
        _, self.images, self.labels, self.traces, _ = trace_data.get_my_teacher()
        #print(self.labels)
        self.length = len(self.images)
        self.create_teacher()


    def get_test(self, even=None):
        """Generate and get train images and traces."""

        self.get_train(even)


    def load_sample(self):
        """Load the sample set and traces."""

        self.load_images(self.folder + "/sampleSet.txt")
        self.load_traces(self.folder + "/sampleLabel.txt")


    def load_images(self, filename):
        """Load the image data"""

        self.images = self.load(filename)
        self.length = len(self.images)
        self.create_teacher()


    def load_traces(self, filename):
        """Load the image data"""

        self.traces = self.load(filename)


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


    def get_explode_length(self):
        """Return the number of traces of all the images combined."""

        return self.explode_length


    def next_batch(self, batch_size):
        """Returns a batch of size batch_size of data."""

        all_idx = np.arange(0, self.length)
        np.random.shuffle(all_idx)
        batch_idx = all_idx[:batch_size]
        batch_imgs = [self.images[i] for i in batch_idx]
        batch_traces = [self.traces[i] for i in batch_idx]
        return batch_imgs, batch_traces


    def next_explode_batch(self, batch_size):
        """Returns a batch of size batch_size of data."""

        all_idx = np.arange(0, self.explode_length)
        np.random.shuffle(all_idx)
        batch_idx = all_idx[:batch_size]
        batch_imgs = [self.explode_images[i] for i in batch_idx]
        batch_lbls = [self.explode_lbls[i] for i in batch_idx]
        batch_labels = [self.explode_labels[i] for i in batch_idx]
        batch_counts = [self.explode_counts[i] for i in batch_idx]
        batch_traces = [self.explode_traces[i] for i in batch_idx]
        return batch_imgs, batch_lbls, batch_labels, batch_counts, batch_traces


    def print_img_at_idx(self, idx):
        """Prints the image at index idx."""

        img = self.images[idx]
        print_img(img)


    def get_trace(self, idx):
        """Returns the trace."""

        return self.traces[idx]


    def create_teacher(self):
        """Create target tensor with fixation positions at each timestep."""

        #words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

        #print("self.length: ", self.length)
        for i, image in enumerate(self.images):
            if False:#i % 2 == 0:
                img = list(chunks(image, 10))
                plt.imshow(img, interpolation="nearest", origin="upper")
                plt.colorbar()
                plt.title(self.labels[i])
                plt.show()
            label = np.argmax(self.labels[i]) + 1
            label_vector = self.labels[i]
            timesteps = label + 1

            for chain in list(self.traces[i]):
                count_tensor = list()
                input_tensor = list()
                target_tensor = list()
                
                #count_tensor.append([0] * z_size)
                #input_tensor.append(image)
                #target_tensor.append([None, None])

                #count_padding = [0 for x in range(z_size)]
                count_end = [1 if x is -1 else 0 for x in range(-1, z_size)]
                
                for count, link in enumerate(chain):
                    count_vector = [0 if x is not count else 1 for x in range(-1, z_size)]
                    count_tensor.append(count_vector)

                    # Mark teacher pointer
                    #x, y = link
                    #image[y*200+x] = 255

                    input_tensor.append(image)
                    target_tensor.append(list(link))

                # Fill in the rest of the list with the same
                #for cont in range(count + 1, z_size): # try this?
                for cont in range(count + 1, z_size + 1):
                    #count_tensor.append(count_padding)
                    count_tensor.append(count_end)
                    #count_tensor.append(count_vector)
                    input_tensor.append(image)
                    target_tensor.append(list(link))

                

                self.explode_lbls.append(label)
                self.explode_labels.append(label_vector)
                self.explode_counts.append(count_tensor)
                self.explode_images.append(input_tensor)
                self.explode_traces.append(target_tensor)
                #print(target_tensor)

        
        
        self.explode_length = len(self.explode_images)


    def explode(self):
        """Create the chain of explode_images and explode_traces formed by counting across two timesteps."""

        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

        for i, image in enumerate(self.images):
            for j, chain in enumerate(list(self.traces[i])):
                for k, link in enumerate(chain):

                    if k is 0:
                        explode_image = image[:]

                        explode_image = np.append(explode_image, ["zero", 0, 0])
                    
                    if k is not len(chain) - 1:
                        next_image = explode_trace = explode_image[:-3] # remove word, x_coor, and y_coor

                        word = words[k]
                        x_coor = link[0]
                        y_coor = link[1]

                        np.append(explode_trace, [word, x_coor, y_coor])

                        # Add the current and the next frames to the data
                        self.explode_images.append(explode_image)
                        self.explode_traces.append(explode_trace)
                        explode_image = next_image
                    else:
                        # No action is made by the teacher after counting is done
                        explode_trace = explode_image
                        self.explode_images.append(explode_image)
                        self.explode_traces.append(explode_trace)

        self.explode_length = len(self.explode_images)


    def explode_chained(self):
        """Create the chain of explode_images and explode_traces formed by counting across all timesteps."""

        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

        for i, image in enumerate(self.images):
            for j, chain in enumerate(list(self.traces[i])):

                # initialize new lists to keep track of another chain
                explode_image_chain = list()
                explode_trace_chain = list()

                for k, link in enumerate(chain):

                    if k is 0:
                        explode_image = image[:]
                        explode_image = np.append(explode_image, ["zero", 0, 0])
                    
                    if k is not len(chain) - 1:
                        next_image = explode_trace = explode_image[:-3]

                        word = words[k]
                        x_coor = link[0]
                        y_coor = link[1]

                        np.append(explode_trace, [word, x_coor, y_coor])

                        # Label the image with the action of the teacher given the image as input
                        explode_image_chain.append(explode_image)
                        explode_trace_chain.append(explode_trace)
                        explode_image = next_image
                    else:
                        explode_trace = explode_image
                        explode_image_chain.append(explode_image)
                        explode_trace_chain.append(explode_trace)

                # Add entire counting sequence grouped together from start to finish
                self.explode_traces.append(explode_trace_chain)
                self.explode_images.append(explode_image_chain)

        self.explode_length = len(self.explode_images)



def test_this():
    """Test out this class."""
    myData = Teacher()
    myData.get_train()
    print("myData.get_length(): ", myData.get_length())
    print("myData.get_explode_length(): ", myData.get_explode_length())
    x_train, explode_lbls, explode_labels, explode_counts, y_train = myData.next_explode_batch(3)
    for i, img in enumerate(x_train):
        print("\n")

        print("new image=================================\n")
        for f, frame in enumerate(img):
            print(frame)
            print("position: ", y_train[i][f])
            print("lbl: ", explode_lbls[i])
            print("label: ", explode_labels[i])
            print("count: ", explode_counts[i][f])
            print("\n")
            #print_img(frame)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_img(img):
    """Prints the image."""
    matrix = list(chunks(img, 10))
    plt.imshow(matrix, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()

