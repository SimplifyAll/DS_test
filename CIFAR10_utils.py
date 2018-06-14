import os
import pickle
import tarfile
import numpy as np
from skimage import img_as_float32
from urllib.request import urlretrieve

class CIFAR10_loader:
    '''
    CIFAR10 dataset loader.

    Only parameter is path to the data.
    Files are assumed to be downloaded from
    https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    and extracted in the given location.
    '''

    IMG_SIZE = 32
    NUM_CHANNELS = 3
    NUM_TRAIN_FILES = 5
    NUM_TEST_FILES = 1
    IMG_PER_FILE = 10000
        # self.total_train_size = self.num_train_files * self.img_per_file

    def __init__(self, data_path):
        self.data_path = data_path
        # self.img_size = 32
        # self.num_channels = 3
        # # self.num_classes = 10
        # self.num_train_files = 5
        # self.num_test_files = 1
        # self.img_per_file = 10000
        # self.total_train_size = self.num_train_files * self.img_per_file

    def load_metadata(self, filename="batches.meta"):
        path = os.path.join(self.data_path, filename)
        with open(path, 'rb') as file:
            meta = pickle.load(file, encoding='bytes')
            return {i: name.decode('utf-8') for i, name in enumerate(meta[b'label_names'])}

    def load_data(self, filename):
        path = os.path.join(self.data_path, filename)
        with open(path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            images = img_as_float32(data[b'data'])
            labels = np.array(data[b'labels'])
            images = images.reshape([images.shape[0], self.NUM_CHANNELS, self.IMG_SIZE, self.IMG_SIZE])\
                    .transpose([0, 2, 3, 1])
            return images, labels

    def load_training_data(self):
        total_train_size = self.NUM_TRAIN_FILES * self.IMG_PER_FILE
        images = np.empty([total_train_size, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS], dtype=np.float32)
        labels = np.empty([total_train_size], dtype=np.int)
        begin = 0
        for i in range(self.NUM_TRAIN_FILES):
            images_batch, labels_batch = self.load_data(filename=f"data_batch_{i + 1}")
            end = begin + self.IMG_PER_FILE
            images[begin:end, :] = images_batch
            labels[begin:end] = labels_batch
            begin = end
            
        # one_hot_labels = LabelBinarizer().fit_transform(labels)
        return images, labels #, one_hot_labels


    def load_test_data(self):
        images, labels = self.load_data(filename="test_batch")
        # one_hot_labels = LabelBinarizer().fit_transform(labels)
        return images, labels #, one_hot_labels


class CIFAR10_downloader:

    DATA_ARCH = "cifar-10-python.tar.gz"
    DATA_DIR = "cifar-10-batches-py"
    DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    def __init__(self, data_path):
        self.data_path = data_path

    
    def extract_tar_gz(self):
        cwd = os.getcwd()
        os.chdir(self.data_path)
        tar = tarfile.open(self.DATA_ARCH, "r:gz")
        tar.extractall()
        tar.close()
        os.chdir(cwd)
    
    def download_cifar(self):
        urlretrieve(self.DATA_URL)
        self.extract_tar_gz()

    def maybe_download(self):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            self.download_cifar()
        elif not os.path.exists(os.path.join(self.data_path, self.DATA_ARCH)):
            self.download_cifar()
        elif not os.path.exists(os.path.join(self.data_path, self.DATA_DIR)):
            self.extract_tar_gz()