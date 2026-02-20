from os.path import join as pjoin
import os


class Config:
    """
    A configuration class for UI element detection and processing in mobile application screenshots.
    
        This class manages the setup and configuration of machine learning models (CNN and EAST) used for
        detecting and classifying UI components, as well as organizing input/output directories for data
        processing pipelines. It provides centralized access to model paths, element classifications,
        visualization colors, and directory structures needed for UI analysis workflows.
    
        Methods:
        - __init__
        - build_output_folders
    
        Attributes:
        - image_shape
        - CNN_PATH
        - element_class
        - class_number
        - EAST_PATH
        - COLOR
        - ROOT_INPUT
        - ROOT_OUTPUT
        - ROOT_IMG_ORG
        - ROOT_IP
        - ROOT_OCR
        - ROOT_MERGE
        - ROOT_IMG_COMPONENT
    """


    def __init__(self):
        """
        Initialize the configuration for UI component detection and visualization.
        
        This constructor establishes the detection pipeline by configuring pre-trained models
        and visual mappings necessary for identifying and highlighting UI elements in mobile
        application screenshots. The CNN model enables classification of interactive components,
        while the EAST model supports text localization for comprehensive UI analysis.
        
        Args:
            None
        
        Attributes:
            image_shape (tuple): Input image dimensions (64, 64, 3) required by the CNN model.
            CNN_PATH (str): File path to the pre-trained CNN model for UI element classification.
            element_class (list): UI element class names detectable by the CNN model: Button, CheckBox,
                Chronometer, EditText, ImageButton, ImageView, ProgressBar, RadioButton, RatingBar,
                SeekBar, Spinner, Switch, ToggleButton, VideoView, and TextView.
            class_number (int): Total count of UI element classes (15).
            EAST_PATH (str): File path to the pre-trained EAST model for text detection.
            COLOR (dict): BGR color mappings for each UI element class and special categories
                (NonText, Compo, Text, Block) used for visual highlighting and overlay rendering.
        
        Returns:
            None
        """
        # setting CNN (graphic elements) model
        self.image_shape = (64, 64, 3)
        # self.MODEL_PATH = 'E:\\Mulong\\Model\\UI2CODE\\cnn6_icon.h5'
        # self.class_map = ['button', 'input', 'icon', 'img', 'text']
        self.CNN_PATH = 'E:/Mulong/Model/rico_compos/cnn-rico-1.h5'
        self.element_class = ['Button', 'CheckBox', 'Chronometer', 'EditText', 'ImageButton', 'ImageView',
                              'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar', 'Spinner', 'Switch',
                              'ToggleButton', 'VideoView', 'TextView']
        self.class_number = len(self.element_class)

        # setting EAST (ocr) model
        self.EAST_PATH = 'E:/Mulong/Model/East/east_icdar2015_resnet_v1_50_rbox'

        self.COLOR = {'Button': (0, 255, 0), 'CheckBox': (0, 0, 255), 'Chronometer': (255, 166, 166),
                      'EditText': (255, 166, 0),
                      'ImageButton': (77, 77, 255), 'ImageView': (255, 0, 166), 'ProgressBar': (166, 0, 255),
                      'RadioButton': (166, 166, 166),
                      'RatingBar': (0, 166, 255), 'SeekBar': (0, 166, 10), 'Spinner': (50, 21, 255),
                      'Switch': (80, 166, 66), 'ToggleButton': (0, 66, 80), 'VideoView': (88, 66, 0),
                      'TextView': (169, 255, 0), 'NonText': (0,0,255),
                      'Compo':(0, 0, 255), 'Text':(169, 255, 0), 'Block':(80, 166, 66)}

    def build_output_folders(self):
        """
        Establishes the directory structure required for processing UI screenshots and detecting interface elements.
        
        This method configures the data flow paths for input datasets and output results across multiple
        processing stages. It initializes paths for storing original images, intermediate processing results
        from image analysis, OCR-extracted text data, merged detection outputs, and isolated UI component
        images. The method ensures all necessary output directories exist, creating them if needed.
        
        Args:
            None
        
        Returns:
            None. This method initializes instance path attributes and creates output directories
            as a side effect.
        
        Attributes Initialized:
            ROOT_INPUT (str): Path to the root input directory containing original UI screenshots.
            ROOT_OUTPUT (str): Path to the root output directory for all processing results.
            ROOT_IMG_ORG (str): Path to the directory containing organized input images.
            ROOT_IP (str): Path to the directory storing image processing and element detection results.
            ROOT_OCR (str): Path to the directory storing OCR text extraction results.
            ROOT_MERGE (str): Path to the directory storing merged detection outputs from multiple methods.
            ROOT_IMG_COMPONENT (str): Path to the directory storing extracted UI component images.
        """
        # setting data flow paths
        self.ROOT_INPUT = "E:\\Mulong\\Datasets\\rico\\combined"
        self.ROOT_OUTPUT = "E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_v3"

        self.ROOT_IMG_ORG = pjoin(self.ROOT_INPUT, "org")
        self.ROOT_IP = pjoin(self.ROOT_OUTPUT, "ip")
        self.ROOT_OCR = pjoin(self.ROOT_OUTPUT, "ocr")
        self.ROOT_MERGE = pjoin(self.ROOT_OUTPUT, "merge")
        self.ROOT_IMG_COMPONENT = pjoin(self.ROOT_OUTPUT, "components")
        if not os.path.exists(self.ROOT_IP):
            os.mkdir(self.ROOT_IP)
        if not os.path.exists(self.ROOT_OCR):
            os.mkdir(self.ROOT_OCR)
        if not os.path.exists(self.ROOT_MERGE):
            os.mkdir(self.ROOT_MERGE)
