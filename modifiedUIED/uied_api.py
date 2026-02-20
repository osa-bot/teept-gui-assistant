from os.path import join as pjoin
import cv2
import os
import numpy as np


def resize_height_by_longest_edge(image, resize_length=800):
    """
    Calculates the resized height based on the longest edge of an image.
    
    This method ensures consistent image processing by normalizing the longest edge to a 
    specified dimension while preserving aspect ratio. This is essential for standardizing 
    input images before analysis, allowing the system to work with images of varying sizes 
    and orientations. If the image's height is greater than its width (portrait orientation),
    it returns the specified resize length. Otherwise, it calculates a proportional height
    based on the aspect ratio.
    
    Args:
        image: Either a file path to an image or a numpy array representing the image.
        resize_length: The target dimension for the longest edge of the image. Defaults to 800.
    
    Returns:
        int: The calculated height value for resizing the image while preserving aspect ratio.
    """
    if isinstance(image, str):  # Если передан путь к файлу
        image = cv2.imread(image)
    height, width = image.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def color_tips():
    """
    Displays a visual reference guide for UI component detection categories.
    
    This method creates and displays a color-coded legend that helps users understand
    how different UI component types are visually represented during analysis. The guide
    generates a 200x200 pixel image divided into four colored sections, each labeled with
    its corresponding component category. This visual reference is essential for users to
    interpret the grid-based overlays and detection results shown during task execution.
    
    Args:
        None
    
    Returns:
        None. Displays a window titled 'colors' containing the color reference guide with
        four component categories: Text (blue), Non-text Components (green), 
        Component Text Content (magenta), and Blocks (cyan).
    """
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 255), 'Text Content': (255, 0, 255)}
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('colors', board)


def run_uied(input_img, output_root='data/output'):
    """
    Executes the complete UI element detection pipeline to identify and extract interactive components from a screenshot.
    
    This method orchestrates a multi-stage detection workflow that identifies text regions, detects UI components,
    and merges overlapping elements to produce a unified representation of the screen's interactive structure.
    The detected elements and their properties enable downstream matching and interaction with UI components
    during task execution.
    
    Args:
        input_img: Either a file path to an image (string) or an image object (numpy array).
        output_root: The root directory path where output results will be saved. Defaults to 'data/output'.
    
    Returns:
        A tuple containing three elements:
        - res: The final merged detection results containing detected UI elements with unified properties.
        - components: The detected UI components from the component detection stage.
        - img_resize: The resized image used during processing.
    """
    # Define or import the classifier here
    classifier = None  # Initialize or load your classifier
    if isinstance(input_img, str):  # Check if the input is a file path or image
        image = cv2.imread(input_img)
    else:
        image = input_img  # If an image object is passed

    key_params = {
        'min-grad': 10,
        'ffl-block': 5,
        'min-ele-area': 50,
        'merge-contained-ele': True,
        'merge-line-to-paragraph': False,
        'remove-bar': True
    }
    
    resized_height = resize_height_by_longest_edge(image, resize_length=800)
    #color_tips()

    import detect_text.text_detection as text
    text_json = text.text_detection(image, show=False, method='paddle')

    import detect_compo.ip_region_proposal as ip
    compo_json = ip.compo_detection(image, key_params,
                                    classifier=classifier, resize_by_height=resized_height, show=False)

    import detect_merge.merge as merge
    res, components, img_resize = merge.merge(image, compo_json, text_json,
                                  is_remove_bar=key_params['remove-bar'], 
                                  is_paragraph=key_params['merge-line-to-paragraph'], 
                                  show=False)

    return res, components, img_resize
