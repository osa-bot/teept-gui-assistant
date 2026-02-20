import os
import pandas as pd
import json
from os.path import join as pjoin
import time
import cv2


def save_corners(file_path, corners, compo_name, clear=True):
    """
    Persists UI component boundary coordinates to a CSV file for task automation reference.
    
    This method enables the system to store detected UI element locations by reading an existing
    CSV file (or creating a new one if it doesn't exist), processing corner coordinate data for
    a specified component, calculating derived spatial dimensions, and writing the updated data
    back to the CSV file. This persistent storage allows the automation system to reference
    previously detected UI element positions across multiple task execution cycles.
    
    Args:
        file_path: The path to the CSV file where corner data will be saved.
        corners: A list of corner coordinate tuples, where each tuple contains
            two points (up_left, bottom_right) representing the top-left and
            bottom-right corners of a component.
        compo_name: The name or identifier of the component being saved.
        clear: Boolean flag indicating whether to clear existing data before saving.
            Defaults to True.
    
    Returns:
        None. The method writes the corner data directly to the specified CSV file.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
    except:
        df = pd.DataFrame(columns=['component', 'x_max', 'x_min', 'y_max', 'y_min', 'height', 'width'])

    if clear:
        df = df.drop(df.index)
    for corner in corners:
        (up_left, bottom_right) = corner
        c = {'component': compo_name}
        (c['y_min'], c['x_min']) = up_left
        (c['y_max'], c['x_max']) = bottom_right
        c['width'] = c['y_max'] - c['y_min']
        c['height'] = c['x_max'] - c['x_min']
        df = df.append(c, True)
    df.to_csv(file_path)


def save_corners_json(file_path, compos):
    """
    Serializes UI component detection results to JSON for task automation and visual guidance.
    
    This method exports detected UI components with their precise bounding box coordinates,
    dimensions, and identifiers to a JSON file. By persisting component metadata including
    image dimensions and spatial properties, this enables the task automation system to
    reference component locations for subsequent interaction planning and visual feedback
    generation through grid-based overlays.
    
    Args:
        file_path (str): The file path where the JSON output will be written.
        compos (list): A list of component objects, each containing spatial layout information
            (bounding box coordinates, dimensions) and identification metadata (id, category).
    
    Returns:
        None. Writes serialized component data directly to the specified file in JSON format.
    """
    img_shape = compos[0].image_shape
    output = {'img_shape': img_shape, 'compos': []}
    f_out = open(file_path, 'w')

    for compo in compos:
        c = {'id': compo.id, 'class': compo.category}
        (c['column_min'], c['row_min'], c['column_max'], c['row_max']) = compo.put_bbox()
        c['width'] = compo.width
        c['height'] = compo.height
        output['compos'].append(c)

    json.dump(output, f_out, indent=4)


def save_clipping(org, output_root, corners, compo_classes, compo_index):
    """
    Saves image clippings of detected UI components to organized directories for training and analysis.
    
    This method extracts rectangular regions from a screenshot based on provided corner coordinates,
    applies padding to capture context around detected components, and saves each clipping as a PNG file
    organized by component class type. This enables systematic collection and categorization of UI elements
    for model training and visual analysis purposes.
    
    Args:
        org: The original image array (screenshot) from which clippings will be extracted.
        output_root: The root directory path where component subdirectories and clippings will be saved.
        corners: A list of corner coordinate pairs, where each pair contains upper-left and bottom-right coordinates defining the bounding box of each component.
        compo_classes: A list of component class names (e.g., 'button', 'text_field') corresponding to each corner region.
        compo_index: A dictionary tracking the index count for each component class type, used to generate unique filenames.
    
    Returns:
        None. The method modifies the compo_index dictionary in-place and writes PNG files to disk organized in class-specific subdirectories.
    """
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    pad = 2
    for i in range(len(corners)):
        compo = compo_classes[i]
        (up_left, bottom_right) = corners[i]
        (col_min, row_min) = up_left
        (col_max, row_max) = bottom_right
        col_min = max(col_min - pad, 0)
        col_max = min(col_max + pad, org.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, org.shape[0])

        # if component type already exists, index increase by 1, otherwise add this type
        compo_path = pjoin(output_root, compo)
        if compo_classes[i] not in compo_index:
            compo_index[compo_classes[i]] = 0
            if not os.path.exists(compo_path):
                os.mkdir(compo_path)
        else:
            compo_index[compo_classes[i]] += 1
        clip = org[row_min:row_max, col_min:col_max]
        cv2.imwrite(pjoin(compo_path, str(compo_index[compo_classes[i]]) + '.png'), clip)


def build_directory(directory):
    """
    Ensures a directory exists for storing task-related data and artifacts.
    
    This method verifies whether the specified directory path exists, creating it if necessary.
    This is essential for the system to have reliable storage locations for screenshots, 
    analysis results, and other workflow artifacts that are generated during task execution.
    The method then returns the directory path for use in subsequent operations.
    
    Args:
        directory (str): The path of the directory to create or verify.
    
    Returns:
        str: The directory path that was created or already existed.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory
