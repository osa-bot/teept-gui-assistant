import json
import cv2
import numpy as np
from os.path import join as pjoin
import os
import time
import shutil

from detect_merge.Element import Element


def show_elements(org_img, eles, show=False, win_name='element', wait_key=0, shown_resize=None, line=2):
    """
    Visualizes detected UI elements on a screenshot by drawing them with category-specific colors for interactive guidance.
    
    This method renders UI elements onto an image with distinct colors based on their category (Text, Component, Block, etc.),
    enabling users to see which screen regions have been identified and are available for interaction. Each element is drawn
    using its own visualization method with a specified line width, providing clear visual feedback during task execution.
    
    Args:
        org_img: The original screenshot image on which elements will be drawn.
        eles: A list of detected UI elements to be visualized on the image.
        show: Whether to display the resulting image in a window. Defaults to False.
        win_name: The name of the window to display the image in. Defaults to 'element'.
        wait_key: The duration in milliseconds to wait for a key press; 0 means wait indefinitely. Defaults to 0.
        shown_resize: Optional tuple specifying the target size for resizing the output image. Defaults to None.
        line: The line width used when drawing elements on the image. Defaults to 2.
    
    Returns:
        The image with visualized UI elements drawn on it, optionally resized according to the shown_resize parameter.
    """
    color_map = {'Text':(0, 0, 255), 'Compo':(0, 255, 0), 'Block':(0, 255, 0), 'Text Content':(255, 0, 255)}
    img = org_img.copy()
    for ele in eles:
        color = color_map[ele.category]
        ele.visualize_element(img, color, line)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        cv2.imshow(win_name, img_resize)
        cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(win_name)
    return img_resize


def save_elements(output_file, elements, img_shape):
    """
    Saves detected UI elements to a JSON file for task automation reference.
    
    This method serializes a collection of UI elements into a JSON format and writes
    them to a specified output file, enabling the system to maintain a structured record
    of screen components for task guidance and element matching. Each element is converted
    to its wrapped information representation and stored alongside image dimensions,
    allowing the automation system to reference UI component locations and properties
    during task execution.
    
    Args:
        output_file: The file path where the JSON data will be written.
        elements: A collection of UI element objects to be serialized, where each element
            has a wrap_info() method that returns its serialized representation containing
            component properties and coordinates.
        img_shape: The shape/dimensions of the image associated with these elements,
            typically a tuple representing (height, width) for screen coordinate mapping.
    
    Returns:
        A dictionary containing the serialized components with keys 'compos' (list of
        element information dictionaries) and 'img_shape' (the image dimensions tuple).
    """
    components = {'compos': [], 'img_shape': img_shape}
    for i, ele in enumerate(elements):
        c = ele.wrap_info()
        # c['id'] = i
        components['compos'].append(c)
    json.dump(components, open(output_file, 'w'), indent=4)
    return components


def reassign_ids(elements):
    """
    Reassign sequential IDs to a collection of UI elements.
    
    This method iterates through the provided elements and assigns each element
    a new ID based on its position in the collection, starting from 0. This ensures
    that element identifiers remain consistent and sequential after modifications to
    the collection, which is essential for maintaining proper element tracking and
    reference integrity in the UI analysis and interaction workflow.
    
    Args:
        elements: A collection of objects that have an id attribute to be reassigned.
    
    Returns:
        None. The method modifies the elements in-place by updating their id attributes.
    """
    for i, element in enumerate(elements):
        element.id = i


def refine_texts(texts, img_shape):
    """
    Filters and refines detected text objects to ensure reliable UI element identification by applying content length and height constraints.
    
    This method removes noise from OCR-detected text that may interfere with accurate UI element matching and interaction.
    Only text objects with meaningful content and appropriate visual prominence are retained, ensuring that subsequent
    text-based matching against user descriptions operates on high-quality candidates.
    
    Args:
        texts: A collection of text objects detected from the screenshot, each containing text_content and height attributes.
        img_shape: A tuple or array representing the dimensions of the image, where the first element
            is the image height used for calculating relative text height.
    
    Returns:
        A list of refined text objects that pass the filtering criteria (text content length > 1
        and relative height < 7.5% of image height).
    """
    refined_texts = []
    for text in texts:
        # remove potential noise
        if len(text.text_content) > 1 and text.height / img_shape[0] < 0.075:
            refined_texts.append(text)
    return refined_texts


def merge_text_line_to_paragraph(elements, max_line_gap=5):
    """
    Merges adjacent text elements into paragraphs based on vertical proximity.
    
    This method processes a list of UI elements, separating text elements from non-text
    elements. It then iteratively merges text elements that are vertically close to
    each other (within the specified line gap threshold) into single paragraph units.
    The merging process continues until no more text elements can be combined. This
    consolidation helps group related text content together, improving the accuracy
    of UI element detection and interaction by treating logically connected text as
    unified components rather than individual fragments.
    
    Args:
        elements: A list of element objects to be processed and merged.
        max_line_gap: The maximum vertical distance (in pixels) between text elements
            for them to be considered part of the same paragraph. Defaults to 5.
    
    Returns:
        A list containing all non-text elements followed by the merged text elements
        (paragraphs), maintaining the combined result of the merging process.
    """
    texts = []
    non_texts = []
    for ele in elements:
        if ele.category == 'Text':
            texts.append(ele)
        else:
            non_texts.append(ele)

    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                inter_area, _, _, _ = text_a.calc_intersection_area(text_b, bias=(0, max_line_gap))
                if inter_area > 0:
                    text_b.element_merge(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return non_texts + texts


def refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.8):
    '''
    Refine detected UI components by filtering out invalid elements and associating text content with their parent components.
    
    This method ensures that only meaningful UI components are retained for task automation by:
    1. Removing components that are completely contained within text regions (false positives)
    2. Removing components whose area is predominantly covered by text (unreliable for interaction)
    3. Associating text elements with their parent components to enable accurate element identification and interaction guidance
    
    Args:
        compos (list): List of detected non-text UI components with area and intersection calculation methods
        texts (list): List of detected text elements with area and intersection calculation methods
        intersection_bias (tuple): Bias values (x, y) applied when calculating intersection areas between components and text. Defaults to (2, 2)
        containment_ratio (float): Threshold ratio (0-1) for determining if one element is contained within another. Defaults to 0.8
    
    Returns:
        list: Refined list of valid UI components and unassociated text elements, ready for task automation and user guidance
    '''
    elements = []
    contained_texts = []
    for compo in compos:
        is_valid = True
        text_area = 0
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(text, bias=intersection_bias)
            if inter > 0:
                # the non-text is contained in the text compo
                if ioa >= containment_ratio:
                    is_valid = False
                    break
                text_area += inter
                # the text is contained in the non-text compo
                if iob >= containment_ratio and compo.category != 'Block':
                    contained_texts.append(text)
        if is_valid and text_area / compo.area < containment_ratio:
            # for t in contained_texts:
            #     t.parent_id = compo.id
            # compo.children += contained_texts
            elements.append(compo)

    # elements += texts
    for text in texts:
        if text not in contained_texts:
            elements.append(text)
    return elements


def check_containment(elements):
    """
    Establishes hierarchical relationships between UI elements based on spatial containment.
    
    This method analyzes pairs of detected UI elements to determine their spatial containment
    relationships, which is essential for building an accurate representation of the screen's
    structural hierarchy. For each pair of elements, it evaluates their spatial relationship
    with a bias of (2, 2) to account for element boundaries. When containment is detected,
    it updates the parent-child relationships by appending child elements to the parent's
    children list and setting the child's parent_id to reference the parent element. This
    hierarchical structure enables more precise element targeting and interaction guidance.
    
    Args:
        elements: A list of element objects representing detected UI components to analyze
            for containment relationships.
    
    Returns:
        None. The method modifies the elements in-place by updating their parent_id
        attributes and children lists based on detected spatial containment relationships.
    """
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1:
                elements[j].children.append(elements[i])
                elements[i].parent_id = elements[j].id
            if relation == 1:
                elements[i].children.append(elements[j])
                elements[j].parent_id = elements[i].id


def remove_top_bar(elements, img_height):
    """
    Removes elements that appear to be part of a top bar from a list of elements.
    
    This method filters out UI elements positioned near the top of the screen with minimal
    height, which typically represent navigation bars, headers, or other chrome elements
    that should be excluded from task interaction analysis. By removing these non-content
    elements, the method ensures that subsequent UI element detection and matching focuses
    on actionable content areas rather than persistent interface decorations.
    
    Args:
        elements: A list of element objects, each containing positional attributes
            including row_min (top position in pixels) and height (element height in pixels).
        img_height: The height of the image/screen in pixels, used to calculate the
            maximum allowable height threshold for identifying top bar elements.
    
    Returns:
        A new list containing only elements that do not match the top bar criteria
        (elements positioned below row 10 or exceeding 4% of image height are retained).
    """
    new_elements = []
    max_height = img_height * 0.04
    for ele in elements:
        if ele.row_min < 10 and ele.height < max_height:
            continue
        new_elements.append(ele)
    return new_elements


def remove_bottom_bar(elements, img_height):
    """
    Removes bottom bar elements from a list of UI elements.
    
    This method filters out small UI components positioned at the bottom of the interface
    that are typically part of navigation or status bars. By excluding these elements from
    the detection results, the method helps focus on the primary interactive content of the
    interface, improving the accuracy of subsequent UI analysis and interaction guidance.
    
    Args:
        elements: A list of UI element objects to filter.
        img_height: The height of the image or GUI window (used for context but not
            directly applied in filtering logic).
    
    Returns:
        A new list of elements with bottom bar elements removed. Elements are excluded
        if they are positioned below row 750, have a height between 20-30 pixels, and
        have a width between 20-30 pixels.
    """
    new_elements = []
    for ele in elements:
        # parameters for 800-height GUI
        if ele.row_min > 750 and 20 <= ele.height <= 30 and 20 <= ele.width <= 30:
            continue
        new_elements.append(ele)
    return new_elements


def compos_clip_and_fill(clip_root, org, compos):
    """
    Extracts and organizes UI components from a screenshot for task automation workflow.
    
    This method isolates individual UI components detected in a screenshot, saves them as separate
    files organized by component type, and generates a reference background image with extracted
    components filled using surrounding pixel colors. This enables the task automation system to
    maintain a structured repository of UI elements for matching and interaction during guided
    task execution.
    
    Args:
        clip_root (str): The root directory path where clipped components and background image will be saved.
        org (numpy.ndarray): The original screenshot image as a numpy array in BGR format (from OpenCV).
        compos (list): A list of component dictionaries, each containing:
            - 'class' (str): The component type/class name
            - 'position' (dict): Bounding box with keys 'column_min', 'row_min', 'column_max', 'row_max'
            - 'id' (int): Unique component identifier
            Each dictionary is updated in-place with a 'path' field indicating the saved file location.
    
    Returns:
        None. Modifies the compos list in-place by adding 'path' entries for each component.
        Creates the following directory structure under clip_root:
        - Subdirectories for each component class (except 'Background')
        - Individual clipped component images saved as JPG files named by component id
        - A background reference image (bkg.png) with all non-background components filled
          using the most frequent surrounding pixel colors
    """
    def most_pix_around(pad=6, offset=2):
        '''
        determine the filled background color according to the most surrounding pixel
        '''
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
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)

    bkg = org.copy()
    cls_dirs = []
    for compo in compos:
        cls = compo['class']
        if cls == 'Background':
            compo['path'] = pjoin(clip_root, 'bkg.png')
            continue
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo['id']) + '.jpg')
        compo['path'] = c_path
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)

        position = compo['position']
        col_min, row_min, col_max, row_max = position['column_min'], position['row_min'], position['column_max'], position['row_max']
        cv2.imwrite(c_path, org[row_min:row_max, col_min:col_max])
        # Fill up the background area
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), most_pix_around(), -1)
    cv2.imwrite(pjoin(clip_root, 'bkg.png'), bkg)


def merge(image, compo_json, text_json, merge_root=None, is_paragraph=False, is_remove_bar=True, show=False, wait_key=0):
    """
    Merges composition and text detection results into unified UI elements for task automation.
    
    This method combines detected UI components and text elements from separate detection outputs
    into a single coherent set of elements that can be used for task guidance and interaction.
    It handles coordinate alignment, element refinement, and optional post-processing such as
    bar removal and text paragraph merging to ensure accurate UI element representation for
    automated task execution.
    
    Args:
        image: The input image to be processed and resized according to composition shape.
        compo_json: JSON object containing detected UI components with their bounding boxes,
            classes, and image shape information.
        text_json: JSON object containing detected text elements with their bounding boxes,
            content, and image shape information.
        merge_root: Optional root directory path for saving merged results.
        is_paragraph: Boolean flag to enable merging of text lines into paragraphs.
        is_remove_bar: Boolean flag to enable removal of top and bottom bars from elements.
        show: Boolean flag to control visualization of intermediate results.
        wait_key: Integer specifying the wait time in milliseconds for visualization windows.
    
    Returns:
        A tuple containing:
            - board: Visualization board (currently returns 0).
            - components: Dictionary with merged UI elements and their information, containing
              'compos' list with wrapped element data and 'img_shape' with resized image dimensions.
            - img_resize: The resized image matching the composition JSON shape.
    """
    ele_id = 0
    compos = []
    
    # Создание элементов композиций
    for compo in compo_json['compos']:
        element = Element(ele_id, (compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']), compo['class'])
        compos.append(element)
        ele_id += 1
        
    texts = []
    for text in text_json['texts']:
        element = Element(ele_id, (text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text', text_content=text['content'])
        texts.append(element)
        ele_id += 1
        
    if compo_json['img_shape'] != text_json['img_shape']:
        resize_ratio = compo_json['img_shape'][0] / text_json['img_shape'][0]
        for text in texts:
            text.resize(resize_ratio)

    # Проверка оригинальных элементов
    img_resize = cv2.resize(image, (compo_json['img_shape'][1], compo_json['img_shape'][0]))
    #show_elements(img_resize, texts + compos, show=show, win_name='all elements before merging', wait_key=wait_key)

    # Уточнение элементов
    texts = refine_texts(texts, compo_json['img_shape'])
    elements = refine_elements(compos, texts)
    if is_remove_bar:
        elements = remove_top_bar(elements, img_height=compo_json['img_shape'][0])
        elements = remove_bottom_bar(elements, img_height=compo_json['img_shape'][0])
    if is_paragraph:
        elements = merge_text_line_to_paragraph(elements, max_line_gap=7)
        
    reassign_ids(elements)
    check_containment(elements)
    #board = show_elements(img_resize, elements, show=show, win_name='elements after merging', wait_key=wait_key)
    board = 0
    # Сохранение всех объединенных элементов, клипов и пустого фона
    name = 'merged_image'  # Уникальное имя файла можно задать по желанию
    components = {
    'compos': [ele.wrap_info() for ele in elements],
    'img_shape': img_resize.shape
    }
    #print('[Merge Completed] Output: %s' % (pjoin(merge_root, name + '.jpg')))
    
    return board, components, img_resize