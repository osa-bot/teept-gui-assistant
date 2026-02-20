import cv2
import numpy as np
import shutil
import os
from os.path import join as pjoin


def segment_img(org, segment_size, output_path, overlap=100):
    """
    Segments an image into overlapping horizontal strips for processing and saves them as separate files.
    
    This method divides a given image into horizontal segments of specified size with optional
    overlap between consecutive segments. Overlapping segments enable comprehensive analysis of UI
    elements that may span across segment boundaries, ensuring no important details are missed during
    detection and matching. Each segment is saved as a PNG file in the output directory for further
    processing or analysis.
    
    Args:
        org: The input image as a numpy array to be segmented.
        segment_size: The height in pixels of each segment to be created.
        output_path: The directory path where the segmented image files will be saved.
        overlap: The height in pixels of overlap between consecutive segments (default: 100).
    
    Returns:
        None. The method saves segmented image files to the specified output directory.
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    height, width = np.shape(org)[0], np.shape(org)[1]
    top = 0
    bottom = segment_size
    segment_no = 0
    while top < height and bottom < height:
        segment = org[top:bottom]
        cv2.imwrite(os.path.join(output_path, str(segment_no) + '.png'), segment)
        segment_no += 1
        top += segment_size - overlap
        bottom = bottom + segment_size - overlap if bottom + segment_size - overlap <= height else height


def clipping(img, components, pad=0, show=False):
    """
    Extract individual UI components from an image by clipping regions around detected elements.
    
    This method isolates each detected component from the original image, creating separate
    image patches that can be further analyzed or processed. This is essential for examining
    specific UI elements in detail and preparing them for visual analysis or matching operations.
    
    Args:
        img: The original image from which components will be clipped.
        components: List of detected UI component objects, each containing bounding box information.
        pad (int, optional): Padding to expand or shrink the clipping region around each component.
            Positive values expand the region, negative values shrink it. Defaults to 0.
        show (bool, optional): If True, displays each clipped component in a window for visual inspection.
            Defaults to False.
    
    Returns:
        list: A list of clipped image patches, one for each component in the input list.
    """
    clips = []
    for component in components:
        clip = component.compo_clipping(img, pad=pad)
        clips.append(clip)
        if show:
            cv2.imshow('clipping', clip)
            cv2.waitKey()
    return clips


def dissemble_clip_img_hollow(clip_root, org, compos):
    """
    Separates UI components from a screenshot and creates a background layer with component regions removed.
    
    This method extracts individual UI components detected in a screenshot and organizes them by type
    for further analysis or processing. It simultaneously generates a background image with all detected
    component regions made transparent, enabling independent handling of UI elements and their context.
    This separation is essential for analyzing UI structure, understanding component relationships, and
    supporting interactive guidance systems that need to distinguish between foreground elements and
    background content.
    
    Args:
        clip_root (str): The root directory path where clipped components and background will be saved.
            If the directory exists, it will be removed and recreated.
        org (numpy.ndarray): The original screenshot image as a numpy array from which components
            will be extracted.
        compos (list): A list of component objects, each containing category information, bounding box
            coordinates, and clipping functionality.
    
    Returns:
        None. Saves clipped component images to subdirectories organized by category under clip_root,
        and saves a background image with component regions hollowed out (made transparent) as 'bkg.png'
        in the clip_root directory.
    """
    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)
    cls_dirs = []

    bkg = org.copy()
    hollow_out = np.ones(bkg.shape[:2], dtype=np.uint8) * 255
    for compo in compos:
        cls = compo.category
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo.id) + '.jpg')
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)
        clip = compo.compo_clipping(org)
        cv2.imwrite(c_path, clip)

        col_min, row_min, col_max, row_max = compo.put_bbox()
        hollow_out[row_min: row_max, col_min: col_max] = 0

    bkg = cv2.merge((bkg, hollow_out))
    cv2.imwrite(os.path.join(clip_root, 'bkg.png'), bkg)


def dissemble_clip_img_fill(clip_root, org, compos, flag='most'):
    """
    Extracts and organizes UI component clips from a screenshot while generating a cleaned background image.
    
    This method decomposes a screenshot into individual UI component images organized by type,
    enabling efficient analysis and processing of detected interface elements. By isolating components
    and reconstructing a background with intelligently sampled colors, it supports downstream tasks
    that require clean separation between UI elements and their context.
    
    Args:
        clip_root (str): The root directory path where extracted component clips will be saved.
            If the directory exists, it will be removed and recreated.
        org (np.ndarray): The original screenshot image as a numpy array from which components
            will be extracted and isolated.
        compos (list): A list of component objects, each containing category information, id,
            and methods to extract clipping regions and bounding boxes from the original image.
        flag (str): The color sampling strategy for filling component regions in the background image.
            Use 'average' to fill with averaged pixel values from surrounding areas, or 'most' to fill
            with the most frequent pixel value from surrounding areas. Defaults to 'most'.
    
    Returns:
        None. Creates a directory structure at clip_root with subdirectories for each component
        category containing individual component images as .jpg files, and saves a cleaned background
        image (bkg.png) with all components filled according to the specified color strategy.
    """

    def average_pix_around(pad=6, offset=3):
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1

        average = []
        for i in range(3):
            avg_up = np.average(org[up:row_min - offset, left:right, i])
            avg_bot = np.average(org[row_max + offset:bottom, left:right, i])
            avg_left = np.average(org[up:bottom, left:col_min - offset, i])
            avg_right = np.average(org[up:bottom, col_max + offset:right, i])
            average.append(int((avg_up + avg_bot + avg_left + avg_right)/4))
        return average

    def most_pix_around(pad=6, offset=2):
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1

        most = []
        for i in range(3):
            val = np.concatenate((org[up:row_min - offset, left:right, i].flatten(),
                            org[row_max + offset:bottom, left:right, i].flatten(),
                            org[up:bottom, left:col_min - offset, i].flatten(),
                            org[up:bottom, col_max + offset:right, i].flatten()))
            # print(val)
            # print(np.argmax(np.bincount(val)))
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)
    cls_dirs = []

    bkg = org.copy()
    for compo in compos:
        cls = compo.category
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo.id) + '.jpg')
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)
        clip = compo.compo_clipping(org)
        cv2.imwrite(c_path, clip)

        col_min, row_min, col_max, row_max = compo.put_bbox()
        if flag == 'average':
            color = average_pix_around()
        elif flag == 'most':
            color = most_pix_around()
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), color, -1)

    cv2.imwrite(os.path.join(clip_root, 'bkg.png'), bkg)
