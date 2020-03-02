import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import create_incr_data_test


class InputData(object):
    """An object of InputData must have images and images_incr.

    Attributes:
        images: A list of lists of image pixels, for each numerosity.
            dimensions: 9 (number of numerosities) x 100 (number of images/numerosity) x 10000 (number of pixels/image)
        images_incr: A list of images with one additional item for each image in images.
            dimensions: 9 x 100 x 100 (number of images with N+1 blobs per image with N blobs) x 10000 
    """

    def __init__(self, folder="", images=[], images_incr=[]):
        """Return an InputData object."""
        self.folder = folder
        self.images = []
        self.images_incr = []
        self.length = 0


    def get_train(self, min_blobs=1, max_blobs=1):
        """Generate and get train images and labels."""
        self.images, self.images_incr = create_incr_data_test.generate_data(min_blobs, max_blobs)
        self.length = len(self.images[0])


    def get_test(self, min_blobs=1, max_blobs=1):
        """Generate and get test images and labels."""
        self.images, self.images_incr = create_incr_data_test.generate_data(min_blobs, max_blobs)
        self.length = len(self.images[0])


    def get_length(self):
        """Return the number of images."""
        return self.length


    def next_batch(self, batch_size=100):
        """Returns a batch of size batch_size of data."""
        all_idx = np.arange(0, self.length)
        np.random.shuffle(all_idx)
        batch_idx = all_idx[:batch_size]
        batch_imgs = []
        batch_imgs_incr = []
        for n, images in enumerate(self.images):
            batch_imgs.append([images[i] for i in batch_idx])
            batch_imgs_incr.append([self.images_incr[n][i] for i in batch_idx])
        return batch_imgs, batch_imgs_incr


    def print_img_at_idx(self, idx):
        """Prints the image at index idx."""
        img = self.images[idx]
        print_img(img)


    def get_incr_imgs_at_idx(self, idx):
        """Returns the images with an incremented number of blobs from the image at index idx."""
        return self.images_incr[idx]


def test_this():
    """Test out this class."""
    myData = InputData()
    myData.get_test(1, 15)
    print(myData.get_length())
    x_train, x_incr_train = myData.next_batch(1)
    for n, imgs in enumerate(x_train):
        print('numerosity: %d' % (n+1))
        for i, img in enumerate(imgs):
            print_img(img, 'N%d_img%d'%(n+1,i))
            for i_incr, img_incr in enumerate(x_incr_train[n][i][:2]):
                print_img(img_incr, 'N%d_img%d_incr%d'%(n+1,i,i_incr))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_img(img, name=""):
    """Prints the image."""
    matrix = list(chunks(img, 100))
    plt.imshow(matrix, interpolation="nearest", origin="upper")
    plt.colorbar()
    #plt.show()
    plt.savefig('images/' + name + '.png')
    plt.clf()


#test_this()
