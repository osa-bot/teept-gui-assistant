import cv2
from os.path import join as pjoin
import json
import numpy as np

import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.file_utils as file
import detect_compo.lib_ip.Component as Compo
from UIED_config.CONFIG_UIED import Config
C = Config()


def nesting_inspection(org, grey, compos, ffl_block):
    '''
    Detect and extract nested UI components within large container elements using flood-fill segmentation.
    
    This method analyzes large components (height > 50 pixels) to identify smaller nested UI elements
    that may be contained within them. By applying flood-fill based block division, it discovers
    internal structure and component hierarchy, which is essential for accurate UI element mapping
    and interaction guidance in automated task workflows.
    
    Args:
        org: Original image array for nested component detection context
        grey: Greyscale image array for gradient-based flood-fill analysis
        compos: List of detected components to inspect for nesting
        ffl_block: Gradient threshold parameter for flood-fill algorithm controlling sensitivity
                  to intensity changes during block division
    
    Returns:
        List of newly detected nested components extracted from large parent components.
        Parent components that contain non-redundant nested elements are replaced in the
        original compos list, while redundant nested components are excluded from results.
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:
            replace = False
            clip_grey = compo.compo_clipping(grey)
            n_compos = det.nested_components_detection(clip_grey, org, grad_thresh=ffl_block, show=False)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def to_arr(compos):
    """
    Converts a list of detected UI component objects into a serializable dictionary representation.
    
    This method transforms component objects into a structured dictionary format that can be
    easily transmitted and processed by the server for task planning and guidance. It extracts
    spatial information and component metadata to enable the system to match user descriptions
    to detected UI elements and provide accurate visual feedback through grid-based overlays.
    
    Args:
        compos: A list of component objects detected from the screenshot, each containing
            image shape, component id, category/class label, bounding box coordinates,
            width, and height information.
    
    Returns:
        A dictionary containing the image shape and a list of component dictionaries.
        Each component dictionary includes the component id, class/category label, bounding
        box coordinates (column_min, row_min, column_max, row_max), width, and height.
        This structure enables efficient matching of UI elements for task automation and
        interactive guidance.
    """
    img_shape = compos[0].image_shape
    output = {'img_shape': img_shape, 'compos': []}

    for compo in compos:
        c = {'id': compo.id, 'class': compo.category}
        (c['column_min'], c['row_min'], c['column_max'], c['row_max']) = compo.put_bbox()
        c['width'] = compo.width
        c['height'] = compo.height
        output['compos'].append(c)

    return output

def compo_detection(input_img, uied_params, resize_by_height=800, classifier=None, show=False, wai_key=0):
    """
    Detects and extracts UI components from a screenshot to enable task automation and visual guidance.
    
    This method processes an input image to identify and extract UI components that can be matched
    with task instructions and presented to users through visual overlays. It performs image binarization,
    component detection, filtering, merging, and nesting inspection to produce a comprehensive list of
    detected UI elements with their spatial properties and relationships.
    
    The detected components serve as the foundation for matching user-provided descriptions and
    generating interactive grid-based guidance for task completion.
    
    Args:
        input_img: Either a file path string to an image or a numpy array representing a screenshot.
        uied_params: A dictionary containing configuration parameters for component detection, including
            'min-grad' for gradient threshold, 'min-ele-area' for minimum element area, 'merge-contained-ele'
            for merging contained elements, and 'ffl-block' for block recognition settings.
        resize_by_height: The target height in pixels to resize the image to. Defaults to 800.
        classifier: An optional classifier object for component classification. Defaults to None.
        show: A boolean flag indicating whether to display intermediate processing results. Defaults to False.
        wai_key: The wait key value for displaying images, used when show is True. Defaults to 0.
    
    Returns:
        A numpy array representation of the detected UI components, where each row contains information
        about a detected component including its bounding box coordinates, size, and containment relationships.
    """
    # Определяем имя файла (или имя по умолчанию для np.array)
    if isinstance(input_img, str):
        name = input_img.split('/')[-1][:-4] if '/' in input_img else input_img.split('\\')[-1][:-4]
        img, grey = pre.read_img(input_img, resize_by_height)  # This returns img and gray
    else:
        name = "captured_image"
        img = pre.resize_img(input_img, resize_by_height)  # This returns only the resized image
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale manually

    # Create directory for output
    #
    # ip_root = file.build_directory(pjoin(output_root, "ip"))

    # Image binarization
    binary = pre.binarization(img, grad_min=int(uied_params['min-grad']))
    det.rm_line(binary, show=show, wait_key=wai_key)

    # Component detection and filtering
    uicompos = det.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']))
    uicompos = det.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=binary.shape)
    uicompos = det.merge_intersected_compos(uicompos)
    det.compo_block_recognition(binary, uicompos)

    # Check for nested components
    if uied_params['merge-contained-ele']:
        uicompos = det.rm_contained_compos_not_in_block(uicompos)

    # Update component information
    Compo.compos_update(uicompos, img.shape)
    Compo.compos_containment(uicompos)
    uicompos += nesting_inspection(img, grey, uicompos, ffl_block=uied_params['ffl-block'])
    Compo.compos_update(uicompos, img.shape)

    # Draw and save results
   # draw.draw_bounding_box(img, uicompos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '.jpg'), wait_key=wai_key)
    Compo.compos_update(uicompos, img.shape)
    arr = to_arr(uicompos)

    #print("Input: %s Output: %s" % (name, pjoin(ip_root, name + '.json')))
    return arr
