import json
import numpy as np
import cv2
from glob import glob
from os.path import join as pjoin
from tqdm import tqdm


def resize_label(bboxes, d_height, gt_height, bias=0):
    """
    Resizes bounding boxes to match target image dimensions by scaling based on height ratio.
    
    This method adjusts bounding box coordinates to account for differences between detected
    and target image heights, ensuring UI element locations remain accurate when screen
    dimensions change. It computes a scale factor from the height ratio and applies it to
    all coordinates, with optional bias adjustment for fine-tuning element positioning.
    
    Args:
        bboxes: A list of bounding boxes, where each bounding box is a list of
            numeric coordinates [x1, y1, x2, y2, ...] to be scaled.
        d_height: The height of the detected or source image in pixels.
        gt_height: The height of the ground truth or target image in pixels.
        bias: An optional offset value to add to each scaled coordinate after
            scaling. Defaults to 0.
    
    Returns:
        A list of resized bounding boxes with scaled and biased integer coordinates,
        adjusted to match the target image dimensions.
    """
    bboxes_new = []
    scale = gt_height / d_height
    for bbox in bboxes:
        bbox = [int(b * scale + bias) for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def draw_bounding_box(org, corners, color=(0, 255, 0), line=2, show=False):
    """
    Visualizes detected bounding boxes on an image for UI element identification and verification.
    
    This method overlays rectangular outlines on an image to highlight detected UI components,
    enabling visual confirmation of element detection results during the task automation workflow.
    Each bounding box represents a detected UI region that can be matched to user instructions.
    The annotated image can optionally be displayed for real-time verification of detection accuracy.
    
    Args:
        org: The original image on which bounding boxes will be drawn.
        corners: A list of bounding box coordinates where each element contains
            the top-left (x1, y1) and bottom-right (x2, y2) corner coordinates of a box.
        color: The color of the bounding box rectangles in BGR format. Defaults to green (0, 255, 0).
        line: The thickness of the bounding box lines in pixels. Defaults to 2.
        show: A boolean flag indicating whether to display the resulting image
            in a window for visual verification. Defaults to False.
    
    Returns:
        The annotated image with bounding boxes drawn on it, preserving the original image.
    """
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color, line)
    if show:
        cv2.imshow('a', cv2.resize(board, (500, 1000)))
        cv2.waitKey(0)
    return board


def load_detect_result_json(reslut_file_root, shrink=4):
    """
    Loads UI component detection results from JSON files and prepares them for task guidance.
    
    This method reads JSON files containing detected UI components from a specified directory,
    filters out components that are too small or positioned in non-interactive regions (top/bottom margins),
    and reorganizes the data into a structured format for matching user instructions to screen elements.
    The bounding boxes are adjusted inward by a specified margin to focus on the interactive core of each component.
    
    Args:
        reslut_file_root: The root directory path containing JSON files with detection results.
        shrink: The number of pixels to shrink each bounding box inward from all sides. Defaults to 4.
    
    Returns:
        A dictionary where keys are image names and values are dictionaries containing:
        - 'bboxes': A list of bounding box coordinates [column_min, row_min, column_max, row_max]
          adjusted inward by the shrink parameter to exclude component borders.
        - 'categories': A list of component category labels corresponding to each bounding box.
    """
    def is_bottom_or_top(corner):
        column_min, row_min, column_max, row_max = corner
        if row_max < 36 or row_min > 725:
            return True
        return False

    result_files = glob(pjoin(reslut_file_root, '*.json'))
    compos_reform = {}
    print('Loading %d detection results' % len(result_files))
    for reslut_file in tqdm(result_files):
        img_name = reslut_file.split('\\')[-1].split('.')[0]
        compos = json.load(open(reslut_file, 'r'))['compos']
        for compo in compos:
            if compo['column_max'] - compo['column_min'] < 10 or compo['row_max'] - compo['row_min'] < 10:
                continue
            if is_bottom_or_top((compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max'])):
                continue
            if img_name not in compos_reform:
                compos_reform[img_name] = {'bboxes': [[compo['column_min'] + shrink, compo['row_min'] + shrink, compo['column_max'] - shrink, compo['row_max'] - shrink]],
                                           'categories': [compo['category']]}
            else:
                compos_reform[img_name]['bboxes'].append([compo['column_min'] + shrink, compo['row_min'] + shrink, compo['column_max'] - shrink, compo['row_max'] - shrink])
                compos_reform[img_name]['categories'].append(compo['category'])
    return compos_reform


def load_ground_truth_json(gt_file):
    """
    Loads ground truth annotations from a JSON file and organizes them by image.
    
    This method reads a JSON file containing annotated UI components and their properties,
    then processes the annotations to create a dictionary mapping image names to their 
    corresponding bounding boxes, categories, and dimensions. This enables efficient 
    lookup of UI element locations and properties during task execution and element matching.
    
    Args:
        gt_file (str): Path to the JSON file containing ground truth annotations in COCO format.
    
    Returns:
        dict: A dictionary where keys are image names (without extension) and values are
        dictionaries containing:
        - 'bboxes' (list): List of bounding boxes in [col_min, row_min, col_max, row_max] format
        - 'categories' (list): List of category IDs corresponding to each bounding box
        - 'size' (tuple): Tuple of (height, width) for the image
    """
    def get_img_by_id(img_id):
        for image in images:
            if image['id'] == img_id:
                return image['file_name'].split('/')[-1][:-4], (image['height'], image['width'])

    def cvt_bbox(bbox):
        '''
        :param bbox: [x,y,width,height]
        :return: [col_min, row_min, col_max, row_max]
        '''
        bbox = [int(b) for b in bbox]
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    data = json.load(open(gt_file, 'r'))
    images = data['images']
    annots = data['annotations']
    compos = {}
    print('Loading %d ground truth' % len(annots))
    for annot in tqdm(annots):
        img_name, size = get_img_by_id(annot['image_id'])
        if img_name not in compos:
            compos[img_name] = {'bboxes': [cvt_bbox(annot['bbox'])], 'categories': [annot['category_id']], 'size': size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def eval(detection, ground_truth, img_root, show=True, no_text=False, only_text=False):
    """
    Evaluates detected UI components against ground truth annotations using Intersection over Union (IoU) matching.
    
    This method validates the accuracy of component detection by comparing detected bounding boxes with 
    ground truth annotations across multiple screenshots. It calculates per-image and aggregate precision, 
    recall, and F1 scores to measure detection performance. The evaluation supports selective filtering 
    to assess text-based components separately from structural UI elements, enabling detailed analysis of 
    detection quality across different component types.
    
    Args:
        detection: Dictionary mapping image IDs to detected components, where each component 
            contains 'bboxes' (list of [col_min, row_min, col_max, row_max]) and 'categories' 
            (list of component type labels).
        ground_truth: Dictionary mapping image IDs to ground truth components with 'bboxes' 
            (list of [col_min, row_min, col_max, row_max]), 'categories' (list of category indices), 
            and 'size' (original image height).
        img_root: Path to the root directory containing the screenshot images to be evaluated.
        show: Boolean flag to display per-image evaluation results and visualizations during 
            evaluation. Defaults to True.
        no_text: Boolean flag to exclude text components (TextView, category 14) from evaluation. 
            Defaults to False.
        only_text: Boolean flag to evaluate only text components (TextView, category 14). 
            Defaults to False.
    
    Returns:
        A tuple of three lists (pres, recalls, f1s) containing per-image precision, recall, and F1 scores 
        respectively. Each list contains scores for images that were successfully evaluated, enabling 
        analysis of detection consistency across the dataset.
    """
    def compo_filter(compos, flag):
        if not no_text and not only_text:
            return compos
        compos_new = {'bboxes': [], 'categories': []}
        for k, category in enumerate(compos['categories']):
            if only_text:
                if flag == 'det' and category != 'TextView':
                    continue
                if flag == 'gt' and int(category) != 14:
                    continue
            elif no_text:
                if flag == 'det' and category == 'TextView':
                    continue
                if flag == 'gt' and int(category) == 14:
                    continue

            compos_new['bboxes'].append(compos['bboxes'][k])
            compos_new['categories'].append(category)
        return compos_new

    def match(org, d_bbox, gt_bboxes, matched):
        '''
        :param matched: mark if the ground truth component is matched
        :param d_bbox: [col_min, row_min, col_max, row_max]
        :param gt_bboxes: list of ground truth [[col_min, row_min, col_max, row_max]]
        :return: Boolean: if IOU large enough or detected box is contained by ground truth
        '''
        area_d = (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])
        for i, gt_bbox in enumerate(gt_bboxes):
            if matched[i] == 0:
                continue
            area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            col_min = max(d_bbox[0], gt_bbox[0])
            row_min = max(d_bbox[1], gt_bbox[1])
            col_max = min(d_bbox[2], gt_bbox[2])
            row_max = min(d_bbox[3], gt_bbox[3])
            # if not intersected, area intersection should be 0
            w = max(0, col_max - col_min)
            h = max(0, row_max - row_min)
            area_inter = w * h
            if area_inter == 0:
                continue
            iod = area_inter / area_d
            iou = area_inter / (area_d + area_gt - area_inter)
            # if show:
            #     cv2.putText(org, (str(round(iou, 2)) + ',' + str(round(iod, 2))), (d_bbox[0], d_bbox[1]),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if iou > 0.9 or iod == 1:
                matched[i] = 0
                return True
        return False

    amount = len(detection)
    TP, FP, FN = 0, 0, 0
    pres, recalls, f1s = [], [], []
    for i, image_id in enumerate(detection):
        TP_this, FP_this, FN_this = 0, 0, 0
        img = cv2.imread(pjoin(img_root, image_id + '.jpg'))
        d_compos = detection[image_id]
        if image_id not in ground_truth:
            continue
        gt_compos = ground_truth[image_id]

        org_height = gt_compos['size'][0]

        d_compos = compo_filter(d_compos, 'det')
        gt_compos = compo_filter(gt_compos, 'gt')

        d_compos['bboxes'] = resize_label(d_compos['bboxes'], 800, org_height)
        matched = np.ones(len(gt_compos['bboxes']), dtype=int)
        for d_bbox in d_compos['bboxes']:
            if match(img, d_bbox, gt_compos['bboxes'], matched):
                TP += 1
                TP_this += 1
            else:
                FP += 1
                FP_this += 1
        FN += sum(matched)
        FN_this = sum(matched)

        try:
            pre_this = TP_this / (TP_this + FP_this)
            recall_this = TP_this / (TP_this + FN_this)
            f1_this = 2 * (pre_this * recall_this) / (pre_this + recall_this)
        except:
            print('empty')
            continue

        pres.append(pre_this)
        recalls.append(recall_this)
        f1s.append(f1_this)
        if show:
            print(image_id + '.jpg')
            print('[%d/%d] TP:%d, FP:%d, FN:%d, Precesion:%.3f, Recall:%.3f' % (
                i, amount, TP_this, FP_this, FN_this, pre_this, recall_this))
            # cv2.imshow('org', cv2.resize(img, (500, 1000)))
            broad = draw_bounding_box(img, d_compos['bboxes'], color=(255, 0, 0), line=3)
            draw_bounding_box(broad, gt_compos['bboxes'], color=(0, 0, 255), show=True, line=2)

        if i % 200 == 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
            print(
                '[%d/%d] TP:%d, FP:%d, FN:%d, Precesion:%.3f, Recall:%.3f, F1:%.3f' % (i, amount, TP, FP, FN, precision, recall, f1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print('[%d/%d] TP:%d, FP:%d, FN:%d, Precesion:%.3f, Recall:%.3f, F1:%.3f' % (i, amount, TP, FP, FN, precision, recall, f1))
    # print("Average precision:%.4f; Average recall:%.3f" % (sum(pres)/len(pres), sum(recalls)/len(recalls)))

    return pres, recalls, f1s


no_text = True
only_text = False

# detect = load_detect_result_json('E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_cls\\ip')
detect = load_detect_result_json('E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_cls\\merge')
# detect = load_detect_result_json('E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_v3\\merge')
# detect = load_detect_result_json('E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_v3\\ocr')
gt = load_ground_truth_json('E:\\Mulong\\Datasets\\rico\\instances_test.json')
eval(detect, gt, 'E:\\Mulong\\Datasets\\rico\\combined', show=False, no_text=no_text, only_text=only_text)
