import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

from UIED_config.CONFIG import Config
cfg = Config()


class CNN:
    """
    A Convolutional Neural Network (CNN) classifier for image classification tasks.
    
        This class provides functionality to build, train, load, and evaluate CNN models
        for various classification types including Text, Noise, Elements, and Image classifiers.
        It supports transfer learning using pre-trained models, image preprocessing, and
        inference on new images.
    
        Attributes:
            classifier_type: The type of classifier ('Text', 'Noise', 'Elements', or 'Image').
            model_path: The file path to the pre-trained model.
            class_map: A list of class labels for the classifier.
            class_number: The total number of classes in the classifier.
            model: The loaded or built Keras model object.
            image_shape: The expected input image shape for the model.
    
        Methods:
            - __init__: Initializes the CNN classifier with a specified type.
            - build_model: Builds and optionally trains a transfer learning model using ResNet50.
            - train: Trains the model on provided data and saves it to disk.
            - load: Loads a pre-trained classifier model based on the specified type.
            - preprocess_img: Preprocesses an image for model inference.
            - predict: Performs inference on one or more images.
            - evaluate: Evaluates the classifier model on test data and computes performance metrics.
    """

    def __init__(self, classifier_type, is_load=True):
        """
        Initialize a CNN classifier for detecting and classifying UI components in screenshots.
        
        This classifier is essential for the task automation system to identify and categorize
        different types of visual elements on the screen, enabling accurate UI component detection
        and interaction guidance.
        
        Args:
            classifier_type (str): The type of classifier to initialize. Must be one of:
                - 'Text': Classifier for detecting and recognizing text elements
                - 'Noise': Classifier for identifying noise or irrelevant visual artifacts
                - 'Elements': Classifier for detecting UI components and interactive elements
            is_load (bool, optional): Whether to automatically load the pre-trained model for the
                specified classifier type upon initialization. Defaults to True.
        
        Returns:
            None
        """
        self.data = None
        self.model = None

        self.classifier_type = classifier_type

        self.image_shape = (32,32,3)
        self.class_number = None
        self.class_map = None
        self.model_path = None
        self.classifier_type = classifier_type
        if is_load:
            self.load(classifier_type)

    def build_model(self, epoch_num, is_compile=True):
        """
        Constructs a deep learning model for multi-class image classification by leveraging
        pre-trained feature extraction capabilities and fine-tuning with task-specific layers.
        
        This method builds a neural network architecture that combines a pre-trained ResNet50
        backbone (with frozen weights to preserve learned features) with custom dense layers
        for the classification task. The approach enables efficient learning from limited data
        by reusing robust feature representations learned from large-scale image datasets.
        The model can be immediately compiled and trained on the provided dataset, or built
        for later training configuration.
        
        Args:
            epoch_num (int): The number of epochs to train the model for if is_compile is True.
            is_compile (bool, optional): A boolean flag indicating whether to compile and train 
                the model immediately after building it. Defaults to True. If True, the model
                is compiled with categorical crossentropy loss and trained on the training data
                with validation on test data.
        
        Returns:
            None. The method modifies the instance by constructing the model architecture and
            optionally training it. The compiled Keras Model object is stored in self.model,
            ready for inference or further training.
        """
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.image_shape)
        for layer in base_model.layers:
            layer.trainable = False
        self.model = Flatten()(base_model.output)
        self.model = Dense(128, activation='relu')(self.model)
        self.model = Dropout(0.5)(self.model)
        self.model = Dense(15, activation='softmax')(self.model)

        self.model = Model(inputs=base_model.input, outputs=self.model)
        if is_compile:
            self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            self.model.fit(self.data.X_train, self.data.Y_train, batch_size=64, epochs=epoch_num, verbose=1,
                           validation_data=(self.data.X_test, self.data.Y_test))

    def train(self, data, epoch_num=30):
        """
        Trains the convolutional neural network model on the provided dataset and persists it for future inference tasks.
        
        This method configures and trains the CNN model using the supplied training data over the specified number of epochs,
        then saves the trained model to disk to enable reuse without retraining. This persistence mechanism is essential for
        maintaining trained state across different sessions and allowing the model to be loaded for prediction tasks.
        
        Args:
            data: The training dataset containing input samples and corresponding labels for model training.
            epoch_num: The number of complete passes through the training dataset. Defaults to 30.
        
        Returns:
            None
        """
        self.data = data
        self.build_model(epoch_num)
        self.model.save(self.model_path)
        print("Trained model is saved to", self.model_path)

    def load(self, classifier_type):
        """
        Initializes and loads a specialized neural network classifier for detecting specific UI element categories.
        
        This method configures the appropriate pre-trained model and class labels needed to analyze
        and classify different types of UI components within screenshots. By loading the correct
        classifier variant, the system can accurately identify text regions, noise artifacts, UI elements,
        or images—each critical for understanding the structure and content of application interfaces
        during task automation.
        
        Args:
            classifier_type (str): The category of UI component classifier to load. Supported values are:
                - 'Text': Detects text-containing regions
                - 'Noise': Identifies visual noise or irrelevant artifacts
                - 'Elements': Classifies UI elements into semantic categories
                - 'Image': Detects image-based content
        
        Returns:
            None. Initializes the following instance attributes:
                - model_path (str): File path to the pre-trained model weights
                - class_map (list): List of class labels for classification output
                - class_number (int): Total number of classes the model can predict
                - model: Loaded Keras neural network model
                - image_shape (tuple): Expected input dimensions (only for 'Elements' type)
        
        Raises:
            FileNotFoundError: If the model file at model_path does not exist
            ValueError: If an unsupported classifier_type is provided
        """
        if classifier_type == 'Text':
            self.model_path = 'E:/Mulong/Model/rico_compos/cnn-textview-2.h5'
            self.class_map = ['Text', 'Non-Text']
        elif classifier_type == 'Noise':
            self.model_path = 'E:/Mulong/Model/rico_compos/cnn-noise-1.h5'
            self.class_map = ['Noise', 'Non-Noise']
        elif classifier_type == 'Elements':
            # self.model_path = 'E:/Mulong/Model/rico_compos/resnet-ele14-19.h5'
            # self.model_path = 'E:/Mulong/Model/rico_compos/resnet-ele14-28.h5'
            # self.model_path = 'E:/Mulong/Model/rico_compos/resnet-ele14-45.h5'
            self.model_path = cfg.CNN_PATH
            self.class_map = cfg.element_class
            self.image_shape = (64, 64, 3)
        elif classifier_type == 'Image':
            self.model_path = 'E:/Mulong/Model/rico_compos/cnn-image-1.h5'
            self.class_map = ['Image', 'Non-Image']
        self.class_number = len(self.class_map)
        self.model = load_model(self.model_path)
        print('Model Loaded From', self.model_path)

    def preprocess_img(self, image):
        """
        Preprocesses an image for model inference by standardizing its format and values.
        
        This method prepares input images for neural network analysis by resizing to the model's
        expected dimensions, normalizing pixel intensities to a standard range, converting to the
        appropriate data type, and structuring the data with a batch dimension. This standardization
        ensures consistent input across different image sources and sizes, enabling reliable visual
        analysis of UI elements and screen content.
        
        Args:
            image: The input image to be preprocessed (typically a screenshot or UI element image).
        
        Returns:
            A preprocessed image as a numpy array with shape (1, height, width, channels),
            with pixel values normalized to float32 in the range [0, 1].
        """
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255).astype('float32')
        x = np.array([x])
        return x

    def predict(self, imgs, compos, load=False, show=False):
        """
        Classify UI components in images using the trained CNN model.
        
        This method processes a batch of UI element images through the neural network to predict
        their categories (e.g., button, text field, image, etc.), enabling automated understanding
        of interface structure and component types for task guidance and interaction.
        
        Args:
            imgs (list): List of preprocessed UI element images to classify.
            compos (list): List of component objects corresponding to each image, which will be
                           updated with predicted category labels.
            load (bool, optional): If True, loads the model from disk before prediction. Defaults to False.
            show (bool, optional): If True, displays each image and prints the predicted category.
                                  Defaults to False.
        
        Returns:
            None: Updates the category attribute of each component object in compos in-place.
        """
        if load:
            self.load(self.classifier_type)
        if self.model is None:
            print("*** No model loaded ***")
            return
        for i in range(len(imgs)):
            X = self.preprocess_img(imgs[i])
            Y = self.class_map[np.argmax(self.model.predict(X))]
            compos[i].category = Y
            if show:
                print(Y)
                cv2.imshow('element', imgs[i])
                cv2.waitKey()

    def evaluate(self, data, load=True):
        """
        Evaluates the trained model's performance on test data to assess classification accuracy.
        
        This method validates the model's ability to correctly classify unseen data by computing
        key performance metrics. It optionally loads the model, generates predictions on the test
        dataset, constructs a confusion matrix, and calculates precision and recall to measure
        how well the model distinguishes between different classes.
        
        Args:
            data: An object containing test data with X_test (features) and Y_test (labels)
                attributes in one-hot encoded format.
            load: A boolean flag indicating whether to load the classifier model before
                evaluation. Defaults to True.
        
        Returns:
            None. The method prints the confusion matrix and precision/recall metrics
                to the console as side effects.
        """
        if load:
            self.load(self.classifier_type)
        X_test = data.X_test
        Y_test = [np.argmax(y) for y in data.Y_test]
        Y_pre = [np.argmax(y_pre) for y_pre in self.model.predict(X_test, verbose=1)]

        matrix = confusion_matrix(Y_test, Y_pre)
        print(matrix)

        TP, FP, FN = 0, 0, 0
        for i in range(len(matrix)):
            TP += matrix[i][i]
            FP += sum(matrix[i][:]) - matrix[i][i]
            FN += sum(matrix[:][i]) - matrix[i][i]
        precision = TP/(TP+FP)
        recall = TP / (TP+FN)
        print("Precision:%.3f, Recall:%.3f" % (precision, recall))