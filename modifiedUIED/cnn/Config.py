
class Config:
    """
    Manages configuration settings for the application or system.
    
        This class encapsulates configuration parameters and settings required for system operation,
        providing a centralized way to access and manage configuration data.
    
        Class Methods:
        - __init__: Initialize the Config instance with default or provided configuration settings.
    
        Class Attributes:
        - DATA_PATH: Path to the dataset directory containing training/validation data.
        - MODEL_PATH: Path to the pre-trained model file in HDF5 format.
        - class_map: List of class labels for classification.
        - image_shape: Tuple representing the input image dimensions.
        - class_number: Integer count of classification classes.
    """

    def __init__(self):
        """
        Initialize configuration for UI component classification.
        
        This constructor establishes the configuration parameters needed to detect and classify
        UI components within desktop interfaces. By setting up model paths, class definitions, and
        image specifications, it enables the system to analyze screenshots and identify different
        types of UI elements, which is essential for understanding the structure of graphical
        interfaces during task automation.
        
        Args:
            self: Instance reference.
        
        Returns:
            None
        
        Attributes:
            DATA_PATH (str): Path to the dataset directory containing training/validation data.
            MODEL_PATH (str): Path to the pre-trained model file in HDF5 format.
            class_map (list): List of class labels for classification ('Text', 'Non-Text').
            image_shape (tuple): Tuple representing the input image dimensions (height, width, channels).
            class_number (int): Integer count of classification classes, derived from class_map length.
        """
        # cnn 4 classes
        # self.MODEL_PATH = 'E:/Mulong/Model/ui_compos/cnn6_icon.h5'   # cnn 4 classes
        # self.class_map = ['Image', 'Icon', 'Button', 'Input']

        # resnet 14 classes
        # self.DATA_PATH = "E:/Mulong/Datasets/rico/elements-14-2"
        # self.MODEL_PATH = 'E:/Mulong/Model/rico_compos/resnet-ele14.h5'
        # self.class_map = ['Button', 'CheckBox', 'Chronometer', 'EditText', 'ImageButton', 'ImageView',
        #                   'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar', 'Spinner', 'Switch',
        #                   'ToggleButton', 'VideoView', 'TextView']  # ele-14

        self.DATA_PATH = "E:\Mulong\Datasets\dataset_webpage\Components3"

        self.MODEL_PATH = 'E:/Mulong/Model/rico_compos/cnn2-textview.h5'
        self.class_map = ['Text', 'Non-Text']

        self.image_shape = (32, 32, 3)
        self.class_number = len(self.class_map)
