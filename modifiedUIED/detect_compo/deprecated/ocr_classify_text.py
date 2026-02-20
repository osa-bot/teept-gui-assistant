import pytesseract as pyt
import cv2

import lib_ip.ip_draw as draw
from UIED_config.CONFIG_UIED import Config
C = Config()


def is_text(img, min_word_area, show=False):
    """
    Determines if an image contains meaningful text content by analyzing detected text regions.
    
    This method uses Tesseract OCR to identify and measure text regions within an image,
    enabling the system to distinguish between UI elements that are text-based versus
    graphical. By comparing the cumulative area of detected text against a configurable
    threshold, it helps filter and classify screen regions during task automation workflows.
    The method also supports visualization of detected text regions for debugging purposes.
    
    Args:
        img: The input image to analyze for text content.
        min_word_area: The minimum ratio of text area to total image area (0.0-1.0) required
            to classify the image as containing meaningful text.
        show: Optional flag to display detected text regions and area statistics for
            debugging purposes. Defaults to False.
    
    Returns:
        bool: True if the image contains text with area ratio exceeding min_word_area,
            False if no text is detected or the text area ratio is below the threshold.
        int: -1 if an error occurs during OCR processing.
    """
    broad = img.copy()
    area_word = 0
    area_total = img.shape[0] * img.shape[1]

    try:
        # ocr text detection
        data = pyt.image_to_data(img).split('\n')
    except:
        print(img.shape)
        return -1
    word = []
    for d in data[1:]:
        d = d.split()
        if d[-1] != '-1':
            if d[-1] != '-' and d[-1] != '—' and int(d[-3]) < 50 and int(d[-4]) < 100:
                word.append(d)
                t_l = (int(d[-6]), int(d[-5]))
                b_r = (int(d[-6]) + int(d[-4]), int(d[-5]) + int(d[-3]))
                area_word += int(d[-4]) * int(d[-3])
                cv2.rectangle(broad, t_l, b_r, (0,0,255), 1)

    if show:
        for d in word: print(d)
        print(area_word/area_total)
        cv2.imshow('a', broad)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # no text in this clip or relatively small text area
    if len(word) == 0 or area_word/area_total < min_word_area:
        return False
    return True


def text_detection(org, img_clean):
    """
    Detects text regions in an image to identify UI elements for task automation.
    
    This method uses Pytesseract to extract text data from a preprocessed image and 
    identifies word regions that meet specific size criteria. By filtering detected words 
    based on height and width constraints, it extracts valid text bounding boxes that 
    correspond to interactive UI components. These bounding boxes are essential for 
    locating and matching UI elements during task execution.
    
    Args:
        org: The original image object (used as fallback return value).
        img_clean: The preprocessed image to perform text detection on.
    
    Returns:
        A tuple containing:
        - corners_word: A list of tuples representing bounding boxes of detected text regions.
          Each tuple contains ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)).
          Returns an empty list if no valid text regions are found.
        - None: Returned as the second element, or the original image object is returned 
          with None if an exception occurs during text extraction.
    """
    try:
        data = pyt.image_to_data(img_clean).split('\n')
    except:
        return org, None
    corners_word = []
    for d in data[1:]:
        d = d.split()
        if d[-1] != '-1':
            if d[-1] != '-' and d[-1] != '—' and 5 < int(d[-3]) < 40 and 5 < int(d[-4]) < 100:
                t_l = (int(d[-6]), int(d[-5]))
                b_r = (int(d[-6]) + int(d[-4]), int(d[-5]) + int(d[-3]))
                corners_word.append((t_l, b_r))
    return corners_word


# def text_merge_word_into_line(org, corners_word, max_words_gap=C.THRESHOLD_TEXT_MAX_WORD_GAP):
#
#     def is_in_line(word):
#         for i in range(len(lines)):
#             line = lines[i]
#             # at the same row
#             if abs(line['center'][1] - word['center'][1]) < max_words_gap:
#                 # small gap between words
#                 if (abs(line['center'][0] - word['center'][0]) - abs(line['width']/2 + word['width']/2)) < max_words_gap:
#                     return i
#         return -1
#
#     def merge_line(word, index):
#         line = lines[index]
#         # on the left
#         if word['center'][0] < line['center'][0]:
#             line['col_min'] = word['col_min']
#         # on the right
#         else:
#             line['col_max'] = word['col_max']
#         line['row_min'] = min(line['row_min'], word['row_min'])
#         line['row_max'] = max(line['row_max'], word['row_max'])
#         line['width'] = line['col_max'] - line['col_min']
#         line['height'] = line['row_max'] - line['row_min']
#         line['center'] = ((line['col_max'] + line['col_min'])/2, (line['row_max'] + line['row_min'])/2)
#
#     words = []
#     for corner in corners_word:
#         word = {}
#         (top_left, bottom_right) = corner
#         (col_min, row_min) = top_left
#         (col_max, row_max) = bottom_right
#         word['col_min'], word['col_max'], word['row_min'], word['row_max'] = col_min, col_max, row_min, row_max
#         word['height'] = row_max - row_min
#         word['width'] = col_max - col_min
#         word['center'] = ((col_max + col_min)/2, (row_max + row_min)/2)
#         words.append(word)
#
#     lines = []
#     for word in words:
#         line_index = is_in_line(word)
#         # word is in current line
#         if line_index != -1:
#             merge_line(word, line_index)
#         # word is not in current line
#         else:
#             # this single word as a new line
#             lines.append(word)
#
#     corners_line = []
#     for l in lines:
#         corners_line.append(((l['col_min'], l['row_min']), (l['col_max'], l['row_max'])))
#     return corners_line

