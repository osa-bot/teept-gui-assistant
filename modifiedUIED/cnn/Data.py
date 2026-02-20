import cv2
import numpy as np
from os.path import join as pjoin
import glob
from tqdm import tqdms
from Config import Config

cfg = Config()


class Data:
    """
    Manages loading and preprocessing of image data for machine learning models.
    
        The Data class handles the complete workflow of loading image datasets from disk,
        organizing them by class, and preparing them for model training. It provides functionality
        to load raw image files, generate training and testing datasets with proper normalization
        and label encoding, and maintain metadata about the dataset structure.
    
        Methods:
        - __init__: Initialize the Data instance with empty data structures and configuration parameters.
        - load_data: Load image data from disk, organize by class, and optionally resize images.
        - generate_training_data: Prepare and split data into normalized training and testing sets with one-hot encoded labels.
    
        Attributes:
        - data_num: Counter for the total number of data samples loaded.
        - images: List to store image data.
        - labels: List to store corresponding labels for images.
        - X_train: Training feature data.
        - Y_train: Training label data.
        - X_test: Testing feature data.
        - Y_test: Testing label data.
        - image_shape: Shape/dimensions of images from configuration.
        - class_number: Total number of classes from configuration.
        - class_map: Mapping of class indices to class names from configuration.
        - DATA_PATH: Path to the data directory from configuration.
    """

    def __init__(self):
        """
        Initialize the Data instance for managing image datasets.
        
        This constructor sets up the data management system with empty data structures
        and loads configuration parameters necessary for handling image classification
        tasks. It establishes the foundation for loading, processing, and organizing
        image datasets with their corresponding class labels, enabling the system to
        prepare data for training and testing workflows.
        
        Args:
            None
        
        Returns:
            None
        
        Attributes:
            data_num (int): Counter for the total number of data samples loaded.
            images (list): List to store image data arrays.
            labels (list): List to store corresponding class labels for images.
            X_train (None or ndarray): Training feature data (initialized as None, populated later).
            Y_train (None or ndarray): Training label data (initialized as None, populated later).
            X_test (None or ndarray): Testing feature data (initialized as None, populated later).
            Y_test (None or ndarray): Testing label data (initialized as None, populated later).
            image_shape (tuple): Shape/dimensions of images from configuration.
            class_number (int): Total number of classes from configuration.
            class_map (dict): Mapping of class indices to class names from configuration.
            DATA_PATH (str): Path to the data directory from configuration.
        """
        self.data_num = 0
        self.images = []
        self.labels = []
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None

        self.image_shape = cfg.image_shape
        self.class_number = cfg.class_number
        self.class_map = cfg.class_map
        self.DATA_PATH = cfg.DATA_PATH

    def load_data(self, resize=True, shape=None, max_number=1000000):
        """
        Loads image data from disk and prepares it for model training and analysis.
        
        This method reads image files from the data directory, organizing them by class
        to enable the system to learn visual patterns and features. It traverses through
        subdirectories (one per class), loads all PNG images up to a specified maximum
        number, and optionally resizes them to a consistent format. Both the images and
        their corresponding class labels are stored in memory for efficient access during
        model training and UI element detection tasks.
        
        Args:
            resize: Whether to resize images to the specified shape. Defaults to True.
            shape: Target shape for resizing images as a tuple (height, width).
                If None, uses the shape stored in self.image_shape. Defaults to None.
            max_number: Maximum number of images to load per class. Defaults to 1000000.
        
        Returns:
            None. Populates self.images with loaded image data and self.labels with
            corresponding class indices. Updates self.data_num with the total count
            of loaded images.
        """
        # if customize shape
        if shape is not None:
            self.image_shape = shape
        else:
            shape = self.image_shape

        # load data
        for p in glob.glob(pjoin(self.DATA_PATH, '*')):
            print("*** Loading components of %s: %d ***" %(p.split('\\')[-1], int(len(glob.glob(pjoin(p, '*.png'))))))
            label = self.class_map.index(p.split('\\')[-1])  # map to index of classes
            for i, image_path in enumerate(tqdm(glob.glob(pjoin(p, '*.png'))[:max_number])):
                image = cv2.imread(image_path)
                if resize:
                    image = cv2.resize(image, shape[:2])
                self.images.append(image)
                self.labels.append(label)

        assert len(self.images) == len(self.labels)
        self.data_num = len(self.images)
        print('%d Data Loaded' % self.data_num)

    def generate_training_data(self, train_data_ratio=0.8):
        """
        Prepares image and label data for model training by normalizing, encoding, and splitting into training and testing sets.
        
        This method ensures consistent data preparation by reshuffling images and labels with a fixed random seed,
        converting categorical labels to one-hot encoded format for multi-class classification, normalizing pixel 
        values to the [0, 1] range, and partitioning the dataset according to the specified training ratio. 
        Summary statistics are printed to verify the split distribution.
        
        Args:
            train_data_ratio (float): The proportion of data to allocate for training (default is 0.8,
                meaning 80% for training and 20% for testing).
        
        Returns:
            None. Populates the following instance attributes:
                X_train (np.ndarray): Normalized training images with shape (num_train_samples, height, width, channels).
                X_test (np.ndarray): Normalized testing images with shape (num_test_samples, height, width, channels).
                Y_train (np.ndarray): One-hot encoded training labels with shape (num_train_samples, num_classes).
                Y_test (np.ndarray): One-hot encoded testing labels with shape (num_test_samples, num_classes).
        """
        # transfer int into c dimensions one-hot array
        def expand(label, class_number):
            # return y : (num_class, num_samples)
            y = np.eye(class_number)[label]
            y = np.squeeze(y)
            return y

        # reshuffle
        np.random.seed(0)
        self.images = np.random.permutation(self.images)
        np.random.seed(0)
        self.labels = np.random.permutation(self.labels)
        Y = expand(self.labels, self.class_number)

        # separate dataset
        cut = int(train_data_ratio * self.data_num)
        self.X_train = (self.images[:cut] / 255).astype('float32')
        self.X_test = (self.images[cut:] / 255).astype('float32')
        self.Y_train = Y[:cut]
        self.Y_test = Y[cut:]

        print('X_train:%d, Y_train:%d' % (len(self.X_train), len(self.Y_train)))
        print('X_test:%d, Y_test:%d' % (len(self.X_test), len(self.Y_test)))
