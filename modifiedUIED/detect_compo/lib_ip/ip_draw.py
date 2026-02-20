import cv2
import numpy as np
from random import randint as rint
from UIED_config.CONFIG_UIED import Config


C = Config()


def draw_bounding_box_class(org, components, color_map=C.COLOR, line=2, show=False, write_path=None, name='board'):
    """
    Visualize detected UI components on the original image with their classification labels to provide visual feedback during task analysis.
    
    This method overlays bounding boxes and class labels on the original screenshot, enabling users to verify that UI elements have been correctly identified and classified. This visual representation is essential for confirming the accuracy of component detection before proceeding with task automation.
    
    Args:
        org: Original image array to draw bounding boxes on
        components: List of detected components, each containing bbox coordinates in format (column_min, row_min, column_max, row_max)
                    where top_left is (column_min, row_min) and bottom_right is (column_max, row_max)
        color_map: Dictionary mapping component categories to BGR color tuples for visual distinction
        line: Line thickness for bounding box drawing (default: 2)
        show: Whether to display the annotated image in a window (default: False)
        write_path: Optional file path to save the annotated image
        name: Window name for display (default: 'board')
    
    Returns:
        Annotated image array with bounding boxes and class labels drawn for all detected components
    """
    board = org.copy()
    for compo in components:
        bbox = compo.put_bbox()
        board = cv2.rectangle(board, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_map[compo.category], line)
        # board = cv2.putText(board, compo.category, (bbox[0]+5, bbox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[compo.category], 2)
    if show:
        cv2.imshow(name, board)
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def draw_bounding_box(org, components, color=(0, 255, 0), line=2,
                      show=False, write_path=None, name='board', is_return=False, wait_key=0):
    """
    Visualize detected UI components by drawing bounding boxes on the original image.
    
    This method renders rectangular outlines around identified UI elements to provide visual feedback
    during the component detection and analysis process. The visualization helps verify that UI elements
    have been correctly identified and located within the screen capture.
    
    Args:
        org: Original image array (numpy array)
        components: List of detected components, each containing bounding box information
                    accessible via put_bbox() method returning (column_min, row_min, column_max, row_max)
        color: RGB tuple for bounding box line color (default: (0, 255, 0) for green)
        line: Line thickness in pixels (default: 2)
        show: Whether to display the annotated image in a window (default: False)
        write_path: File path to save the annotated image; if None, image is not saved (default: None)
        name: Window name for display (default: 'board')
        is_return: Whether to return the annotated image (default: False)
        wait_key: Milliseconds to wait for keyboard input; 0 means wait indefinitely (default: 0)
    
    Returns:
        Annotated image array with bounding boxes drawn, or None if show, write_path, and is_return are all False
    """
    if not show and write_path is None and not is_return: return
    board = org.copy()
    for compo in components:
        bbox = compo.put_bbox()
        board = cv2.rectangle(board, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line)
    if show:
        cv2.imshow(name, board)
        if wait_key is not None:
            cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(name)
    if write_path is not None:
        # board = cv2.resize(board, (1080, 1920))
        # board = board[100:-110]
        cv2.imwrite(write_path, board)
    return board


def draw_line(org, lines, color=(0, 255, 0), show=False):
    """
    Visualize detected structural lines on the image to provide feedback on UI element boundaries and layout analysis.
    
    This method overlays horizontal and vertical lines onto a copy of the original image, enabling users to see
    the detected structural components that guide task automation and UI element localization.
    
    Args:
        org: Original image as a numpy array (BGR format).
        lines: Tuple of two lists [line_h, line_v] containing detected lines:
            - line_h: List of horizontal lines, each as a dict with keys:
                - 'head': Tuple (column_min, row) marking the line start
                - 'end': Tuple (column_max, row) marking the line end
                - 'thickness': Integer line thickness in pixels
            - line_v: List of vertical lines, each as a dict with keys:
                - 'head': Tuple (column, row_min) marking the line start
                - 'end': Tuple (column, row_max) marking the line end
                - 'thickness': Integer line thickness in pixels
        color: RGB tuple (B, G, R) specifying the line color. Defaults to (0, 255, 0) for green.
        show: Boolean flag to display the result in a window. Defaults to False.
    
    Returns:
        Image with detected lines drawn as a numpy array (same shape as input).
    """
    board = org.copy()
    line_h, line_v = lines
    for line in line_h:
        cv2.line(board, tuple(line['head']), tuple(line['end']), color, line['thickness'])
    for line in line_v:
        cv2.line(board, tuple(line['head']), tuple(line['end']), color, line['thickness'])
    if show:
        cv2.imshow('img', board)
        cv2.waitKey(0)
    return board


def draw_boundary(components, shape, show=False):
    """
    Visualize detected UI component boundaries on a binary image for verification and debugging.
    
    This method renders the detected boundaries of UI components as white lines on a black background,
    enabling visual inspection of how accurately components have been identified and localized within
    the screen or image. This is essential for validating the component detection pipeline before
    proceeding with interaction or analysis tasks.
    
    Args:
        components: List of detected components, each containing boundary information structured as:
                    [top, bottom, left, right] where:
                    - top, bottom: tuples of (column_index, min/max row border) for horizontal edges
                    - left, right: tuples of (row_index, min/max column border) for vertical edges
        shape: Tuple representing the shape of the original image (height, width, channels).
               Used to initialize the output binary image dimensions.
        show: Boolean flag to display the resulting boundary visualization using OpenCV.
              Default is False.
    
    Returns:
        np.ndarray: Binary image (uint8) with shape matching input image height and width,
                    where detected component boundaries are marked with white pixels (255)
                    on a black background (0).
    """
    board = np.zeros(shape[:2], dtype=np.uint8)  # binary board
    for component in components:
        # up and bottom: (column_index, min/max row border)
        for point in component.boundary[0] + component.boundary[1]:
            board[point[1], point[0]] = 255
        # left, right: (row_index, min/max column border)
        for point in component.boundary[2] + component.boundary[3]:
            board[point[0], point[1]] = 255
    if show:
        cv2.imshow('rec', board)
        cv2.waitKey(0)
    return board


def draw_region(region, broad, show=False):
    """
    Visualizes a detected region on an image by overlaying it with a distinct color.
    
    This method marks a region of interest on the image canvas by coloring all its
    constituent points with a randomly generated color. This visual representation
    helps users identify and understand which UI elements or screen areas have been
    detected and analyzed. Optionally displays the annotated result for immediate
    feedback.
    
    Args:
        region: A collection of points representing the region to visualize, where
            each point is a coordinate pair (row, column) in the image.
        broad: A numpy array representing the image or canvas where the region will
            be drawn. The array is modified in-place with the colored region.
        show: A boolean flag indicating whether to display the annotated image
            using OpenCV. Defaults to False.
    
    Returns:
        The modified broad array with the region points colored with a randomly
        generated RGB color for visual distinction.
    """
    color = (rint(0,255), rint(0,255), rint(0,255))
    for point in region:
        broad[point[0], point[1]] = color

    if show:
        cv2.imshow('region', broad)
        cv2.waitKey()
    return broad


def draw_region_bin(region, broad, show=False):
    """
    Marks detected UI region points on a binary image to create a visual representation for grid-based guidance overlay.
    
    This method highlights all points belonging to a detected region by setting their pixels to white (255) in the binary image, enabling the creation of visual feedback overlays that guide users to interact with specific UI components. The resulting marked image serves as a mask for the on-screen grid interface.
    
    Args:
        region: A collection of coordinate pairs (row, column) representing pixel locations of the detected UI region to be highlighted.
        broad: A binary image array (typically a numpy array) where the region will be marked. Modified in-place.
        show: A boolean flag to display the marked region image for debugging purposes. If True, displays the image and waits for a key press. Defaults to False.
    
    Returns:
        The modified binary image array with the region marked in white, ready to be used as a visual guidance overlay.
    """
    for point in region:
        broad[point[0], point[1]] = 255

    if show:
        cv2.imshow('region', broad)
        cv2.waitKey()
    return broad
