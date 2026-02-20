import cv2
import numpy as np
from UIED_config.CONFIG_UIED import Config
C = Config()

def resize_img(img, resize_height):
    """
    Resizes an image while maintaining its aspect ratio for consistent UI element analysis.
    
    This method scales an image to a target height while preserving the original aspect ratio,
    ensuring that UI elements and their spatial relationships remain proportionally accurate
    during screenshot processing and analysis. This is essential for maintaining consistency
    when detecting and locating UI components across different screen resolutions.
    
    Args:
        img: The input image (numpy array) to be resized.
        resize_height: The target height for the resized image in pixels.
    
    Returns:
        numpy.ndarray: The resized image with the specified height and proportionally adjusted width.
    """
    w_h_ratio = img.shape[1] / img.shape[0]
    resize_w = int(resize_height * w_h_ratio)
    resized_img = cv2.resize(img, (resize_w, resize_height))
    return resized_img

def read_img(input_img, resize_height=None, kernel_size=None):
    """
    Loads and preprocesses an image for UI element detection and analysis.
    
    Handles both file paths and image arrays as input, applying optional noise reduction
    and resizing to prepare images for downstream vision-based UI component detection.
    Converts to both color and grayscale formats to support different analysis methods.
    
    Args:
        input_img (str or ndarray): File path to an image or a numpy array representing an image.
        resize_height (int, optional): Target height for resizing the image. If specified, the image
            is resized proportionally based on this height. Defaults to None (no resizing).
        kernel_size (int, optional): Kernel size for median blur filtering to reduce noise.
            Must be an odd integer. Defaults to None (no filtering).
    
    Returns:
        tuple: A tuple containing:
            - img (ndarray or None): The preprocessed color image (BGR format), or None if loading failed.
            - gray (ndarray or None): The grayscale version of the preprocessed image, or None if loading failed.
    """
    def resize_by_height(org):
        return resize_img(org, resize_height)

    try:
        if isinstance(input_img, str):  # Check if input is a file path
            img = cv2.imread(input_img)
            if img is None:
                print("*** Image does not exist ***")
                return None, None
        else:
            img = input_img.copy()

        if kernel_size is not None:
            img = cv2.medianBlur(img, kernel_size)
        
        if resize_height is not None:
            img = resize_by_height(img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    except Exception as e:
        print(e)
        print("*** Image Reading Failed ***\n")
        return None, None



def gray_to_gradient(img):
    """
    Converts a grayscale or color image to its gradient representation for edge detection.
    
    This method computes the image gradient by applying horizontal and vertical
    Sobel-like kernels to detect edges and intensity changes. Gradient computation
    is essential for identifying UI element boundaries and visual features in
    screenshots. If the input is a color image, it is first converted to grayscale
    to simplify processing.
    
    Args:
        img: Input image, either grayscale (2D array) or color (3D array in BGR format).
    
    Returns:
        uint8: A gradient image where pixel values represent the magnitude of intensity
            changes, computed as the sum of absolute horizontal and vertical gradients.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = np.copy(img)
    img_f = img_f.astype("float")

    kernel_h = np.array([[0,0,0], [0,-1.,1.], [0,0,0]])
    kernel_v = np.array([[0,0,0], [0,-1.,0], [0,1.,0]])
    dst1 = abs(cv2.filter2D(img_f, -1, kernel_h))
    dst2 = abs(cv2.filter2D(img_f, -1, kernel_v))
    gradient = (dst1 + dst2).astype('uint8')
    return gradient


def reverse_binary(bin, show=False):
    """
    Invert the binary representation of a UI element image to prepare it for further processing in element detection workflows.
    
    This method inverts the pixel values of a binary image, converting foreground to background and vice versa.
    This is essential for normalizing UI element masks before analysis, ensuring consistent representation
    across different detection and matching operations in the task automation pipeline.
    
    Args:
        bin: Binary image (numpy array) to be inverted, typically a thresholded UI element mask.
        show (bool, optional): If True, displays the inverted binary image in a window for debugging purposes.
            Defaults to False.
    
    Returns:
        numpy.ndarray: The inverted binary image with pixel values reversed (0 becomes 255 and vice versa).
    """
    r, bin = cv2.threshold(bin, 1, 255, cv2.THRESH_BINARY_INV)
    if show:
        cv2.imshow('binary_rev', bin)
        cv2.waitKey()
    return bin


def binarization(org, grad_min, show=False, write_path=None, wait_key=0):
    """
    Converts an image to binary form through gradient-based thresholding and morphological operations to isolate regions of interest.
    
    This method processes an input image to extract and enhance areas with significant visual changes, which is essential for identifying UI elements and interactive components on screen. It converts the image to grayscale, computes its gradient to detect edges and boundaries, applies binary thresholding to isolate high-gradient regions, and performs morphological closing to eliminate noise and create clean, connected regions suitable for further analysis.
    
    Args:
        org (numpy.ndarray): The original input image in BGR color space.
        grad_min (int): The minimum gradient threshold value for binary thresholding. Pixels with gradient values below this threshold are set to 0, while those above are set to 255.
        show (bool, optional): Whether to display the resulting binary image using cv2.imshow. Defaults to False.
        write_path (str, optional): File path where the binary image should be saved. If None, no file is written. Defaults to None.
        wait_key (int, optional): The duration in milliseconds to wait for a key press when displaying the image. Defaults to 0.
    
    Returns:
        numpy.ndarray: A binary image (uint8 array) after thresholding and morphological closing operations, with values of 0 or 255.
    """
    grey = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    grad = gray_to_gradient(grey)        # get RoI with high gradient
    rec, binary = cv2.threshold(grad, grad_min, 255, cv2.THRESH_BINARY)    # enhance the RoI
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (3, 3))  # remove noises
    if write_path is not None:
        cv2.imwrite(write_path, morph)
    if show:
        cv2.imshow('binary', morph)
        if wait_key is not None:
            cv2.waitKey(wait_key)
    return morph
