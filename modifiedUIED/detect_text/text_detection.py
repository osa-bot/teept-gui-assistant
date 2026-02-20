import detect_text.ocr as ocr
from detect_text.Text import Text
import numpy as np
import cv2
import json
import os
from os.path import join as pjoin


def save_detection_json(texts, img_shape):
    """
    Converts detected UI text elements into a JSON-serializable dictionary format for server communication and task tracking.
    
    This method structures text detection results from OCR analysis into a standardized format that can be transmitted to the server and stored for task automation workflows. It organizes detected text objects with their spatial properties into a dictionary containing image metadata and detailed element information, enabling the system to match user-provided descriptions to UI components and maintain accurate element location records.
    
    Args:
        texts: A collection of text detection objects from OCR analysis, each containing id, content,
            location (with left, top, right, bottom coordinates), width, and height attributes.
        img_shape: The shape or dimensions of the analyzed screenshot.
    
    Returns:
        A dictionary with 'img_shape' key containing the image dimensions and 'texts' key containing 
        a list of dictionaries. Each text dictionary includes the text id, content, bounding box 
        coordinates (column_min, row_min, column_max, row_max), width, and height for UI element 
        localization and matching.
    """
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = loc['left'], loc['top'], loc['right'], loc['bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)
    return output

def visualize_texts(org_img, texts, shown_resize_height=None, show=False, write_path=None):
    """
    Visualizes detected text elements on a screenshot with optional display and file output.
    
    This method renders visual annotations for each text element onto a copy of the original
    image, enabling users to see identified UI text components and their locations. This is
    essential for providing real-time visual feedback during task execution, allowing users
    to verify that the correct text elements have been detected before proceeding with
    automated interactions.
    
    Args:
        org_img: The original screenshot image on which text elements will be visualized.
        texts: A collection of detected text objects to be visualized on the image.
        shown_resize_height: The height to which the displayed image should be resized
            for visualization purposes. If not provided, the image is displayed at its
            original size. Defaults to None.
        show: A boolean flag indicating whether to display the visualized image in a
            window for immediate user feedback. Defaults to False.
        write_path: The file path where the visualized image should be saved for
            logging or debugging purposes. If not provided, the image is not written
            to disk. Defaults to None.
    
    Returns:
        None. The method displays and/or saves the annotated image based on the
        provided parameters.
    """
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)

    img_resize = img
    if shown_resize_height is not None:
        img_resize = cv2.resize(img, (int(shown_resize_height * (img.shape[1]/img.shape[0])), shown_resize_height))

    if show:
        cv2.imshow('texts', img_resize)
        cv2.waitKey(0)
        cv2.destroyWindow('texts')
    if write_path is not None:
        cv2.imwrite(write_path, img)

def text_sentences_recognition(texts):
    '''
    Consolidate fragmented text detections into coherent text lines for improved UI element recognition.
    
    This method iteratively merges individual text fragments that appear on the same horizontal line,
    enabling more accurate identification of complete UI labels and text content. By combining spatially
    adjacent text elements, the method produces cleaner text representations that better correspond to
    actual UI components, which is essential for reliable element matching and interaction guidance.
    
    Args:
        texts (list): A list of text detection objects, each containing positional and dimensional
                      attributes (height, word_width) and methods for spatial comparison and merging.
    
    Returns:
        list: A list of merged text objects with updated sequential IDs, where horizontally aligned
              text fragments have been consolidated into single text elements.
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(text_b, 'h', bias_justify=0.2 * min(text_a.height, text_b.height), bias_gap=2 * max(text_a.word_width, text_b.word_width)):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def merge_intersected_texts(texts):
    '''
    Consolidate overlapping text regions by iteratively merging intersected elements.
    
    This method processes a collection of text elements (such as OCR-detected words or sentences)
    and combines those that spatially overlap or are in close proximity. By repeatedly merging
    intersected elements until no further changes occur, it reduces fragmentation and produces
    a cleaner set of consolidated text regions. This is essential for accurate UI element
    detection and text extraction, as OCR and detection methods often produce overlapping
    or adjacent text fragments that need to be unified for proper element identification.
    
    Args:
        texts (list): A list of text objects, each with methods is_intersected() and merge_text()
                      to detect spatial overlap and combine their properties.
    
    Returns:
        list: A consolidated list of text objects with overlapping elements merged together.
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_intersected(text_b, bias=2):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts


def text_cvt_orc_format(ocr_result):
    """
    Converts OCR results into a standardized text format with location information for UI element detection.
    
    This method processes OCR detection results to extract recognized text content and their spatial 
    locations on the screen, enabling the system to match detected UI elements with task-related 
    descriptions. The bounding box coordinates are normalized into a consistent format (left, top, 
    right, bottom) to facilitate accurate UI component identification and interaction guidance.
    
    Args:
        ocr_result: A list of OCR detection results, where each result contains a
            'boundingPoly' with 'vertices' (coordinate points) and a 'description'
            (the detected text). Can be None.
    
    Returns:
        A list of Text objects, each containing an index, the detected text content,
        and its normalized bounding box location. Returns an empty list if ocr_result
        is None or if no valid results are found.
    """
    texts = []
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            error = False
            x_coordinates = []
            y_coordinates = []
            text_location = result['boundingPoly']['vertices']
            content = result['description']
            for loc in text_location:
                if 'x' not in loc or 'y' not in loc:
                    error = True
                    break
                x_coordinates.append(loc['x'])
                y_coordinates.append(loc['y'])
            if error: continue
            location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                        'right': max(x_coordinates), 'bottom': max(y_coordinates)}
            texts.append(Text(i, content, location))
    return texts


def text_cvt_orc_format_paddle(paddle_result):
    """
    Converts OCR results from PaddleOCR format to a standardized text format for UI element detection.
    
    This method processes the output from PaddleOCR and transforms it into a list of Text objects
    containing location information and content. By extracting bounding box coordinates from the
    detection results, it enables the system to map detected text regions to specific screen locations,
    which is essential for identifying and interacting with UI components during task automation.
    The method creates location dictionaries with left, top, right, and bottom values derived from
    the polygon coordinates of each detected text region.
    
    Args:
        paddle_result: The OCR detection results from PaddleOCR, structured as a list of lines,
            where each line contains elements with polygon coordinates and recognized text content.
            Each element is expected to have format [points, [text_content, confidence]].
    
    Returns:
        list: A list of Text objects, each containing the line index, text content, and bounding box
            location information (left, top, right, bottom coordinates) that can be used for
            UI element matching and interaction.
    """
    texts = []
    for i, line in enumerate(paddle_result):
        try:
            for element in line:  # Перебираем каждый элемент строки
                points = np.array(element[0])
                location = {
                    'left': int(min(points[:, 0])),
                    'top': int(min(points[:, 1])),
                    'right': int(max(points[:, 0])),
                    'bottom': int(max(points[:, 1]))
                }
                content = element[1][0]  # Содержание текста
                texts.append(Text(i, content, location))
        except Exception as e:
            print(f"Ошибка при обработке строки {i}: {e}")
    return texts



def text_filter_noise(texts):
    """
    Filters out noise from detected UI text elements to improve matching accuracy.
    
    This method removes text objects that are considered noise, such as isolated
    single characters that are not meaningful UI labels or delimiters. By retaining
    only substantive text content and valid single-character tokens (common
    punctuation and symbols), the method ensures that subsequent UI element matching
    and analysis operations work with clean, relevant text data.
    
    Args:
        texts: A collection of text objects, each having a content attribute
            containing the text string to be evaluated.
    
    Returns:
        A list of text objects that passed the noise filter, excluding those
        with single-character content that is not in the valid tokens list
        (valid tokens: 'a', ',', '.', '!', '?', '$', '%', ':', '&', '+').
    """
    valid_texts = []
    for text in texts:
        if len(text.content) <= 1 and text.content.lower() not in ['a', ',', '.', '!', '?', '$', '%', ':', '&', '+']:
            continue
        valid_texts.append(text)
    return valid_texts
    

def text_detection(input_img='../data/input/30800.jpg', show=False, method='google', paddle_model=None):
    """
    Detects and extracts text regions from an image to identify UI elements and their content for task automation guidance.
    
    This method processes an input image to detect and extract text regions using OCR technology. It supports
    two OCR backends (Google and Paddle) and applies various post-processing steps including text merging, 
    noise filtering, and sentence recognition. The detected text regions are converted to a standardized JSON 
    format that maps text content to spatial coordinates, enabling the system to match user-provided descriptions 
    to UI components and provide accurate visual guidance for task execution.
    
    Args:
        input_img: Path to the input image file or a numpy array representing the image.
            Defaults to '../data/input/30800.jpg'.
        show: Boolean flag indicating whether to display visualization of detected text regions.
            Defaults to False.
        method: OCR detection method to use. Must be either 'google' or 'paddle'.
            Defaults to 'google'.
        paddle_model: Pre-initialized PaddleOCR model instance. If None and method is 'paddle',
            a new model will be created. Defaults to None.
    
    Returns:
        A numpy array containing the detected text regions in standardized JSON format,
        including bounding box coordinates, confidence scores, and recognized text content
        that can be matched against user task descriptions.
    
    Raises:
        ValueError: If the method parameter is not 'google' or 'paddle'.
    """
    if isinstance(input_img, str):  # Если передан путь к файлу
        img = cv2.imread(input_img)
        name = input_img.split('/')[-1][:-4]
    else:  # Если передано изображение в формате numpy array
        img = input_img
        name = 'captured_image'  # Имя по умолчанию

    if method == 'google':
        print('*** Detect Text through Google OCR ***')
        ocr_result = ocr.ocr_detection_google(img if isinstance(input_img, np.ndarray) else input_img)
        texts = text_cvt_orc_format(ocr_result)
        texts = merge_intersected_texts(texts)
        texts = text_filter_noise(texts)
        texts = text_sentences_recognition(texts)
    elif method == 'paddle':
        from paddleocr import PaddleOCR
        print('*** Detect Text through Paddle OCR ***')
        if paddle_model is None:
            paddle_model = PaddleOCR(use_angle_cls=True, lang="cyrillic")
        result = paddle_model.ocr(img if isinstance(input_img, np.ndarray) else input_img, cls=True)
        texts = text_cvt_orc_format_paddle(result)
    else:
        raise ValueError('Method has to be "google" or "paddle"')

    #visualize_texts(img, texts, shown_resize_height=800, show=show, write_path=pjoin(ocr_root, name + '.png'))
    arr = save_detection_json(texts, img.shape)
    #print("Input: %s Output: %s" % (name, pjoin(ocr_root, name + '.json')))
    return arr
