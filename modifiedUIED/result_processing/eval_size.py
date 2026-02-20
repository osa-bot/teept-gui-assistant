import json
import numpy as np
import cv2
from glob import glob
from os.path import join as pjoin
from tqdm import tqdm


def resize_label(bboxes, d_height, gt_height, bias=0):
    """
    Resizes bounding boxes to match ground truth dimensions by applying a height-based scale factor.
    
    This method normalizes bounding box coordinates to account for differences between detected
    and ground truth UI element dimensions. By computing the ratio of ground truth height to
    detected height, it scales all coordinates proportionally, ensuring that detected UI element
    boundaries align with their actual screen positions. This is essential for accurate element
    localization and interaction guidance in the task automation workflow.
    
    Args:
        bboxes: A list of bounding boxes, where each bounding box is a list of
            numeric coordinates representing UI element boundaries to be scaled.
        d_height: The detected height value used as the denominator in the scaling
            ratio calculation.
        gt_height: The ground truth height value used as the numerator in the scaling
            ratio calculation.
        bias: An optional offset value to be added to each scaled coordinate,
            useful for fine-tuning element positions. Defaults to 0.
    
    Returns:
        A list of resized bounding boxes, where each bounding box contains
        integer-valued scaled coordinates adjusted by the bias parameter.
    """
    bboxes_new = []
    scale = gt_height / d_height
    for bbox in bboxes:
        bbox = [int(b * scale + bias) for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def draw_bounding_box(org, corners, color=(0, 255, 0), line=2, show=False):
    """
    Visualizes detected UI element regions by drawing bounding boxes on a screenshot.
    
    This method resizes the input image to a standardized dimension and overlays rectangular
    bounding boxes at the specified corner coordinates. This visual representation helps users
    identify and interact with detected UI elements during task execution. Optionally displays
    the annotated result in a window for immediate verification.
    
    Args:
        org: The original screenshot image to annotate with bounding boxes.
        corners: A list of bounding box coordinates where each element contains the
            corner positions in the format (x1, y1, x2, y2).
        color: The color of the bounding box lines in BGR format. Defaults to green (0, 255, 0).
        line: The thickness of the bounding box lines in pixels. Defaults to 2.
        show: A boolean flag indicating whether to display the annotated image
            in a window. Defaults to False.
    
    Returns:
        The resized image (608x1024) with bounding boxes drawn at the specified coordinates.
    """
    board = cv2.resize(org, (608, 1024))
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color, line)
    if show:
        cv2.imshow('a', board)
        cv2.waitKey(0)
    return board


def load_detect_result_json(reslut_file_root, shrink=3):
    """
    Loads detection results from JSON files and reformats them into a structured dictionary for UI component analysis.
    
    This method reads JSON files containing component detection results from a specified directory,
    filters out components that are too small or located at the top/bottom of the image (outside the interactive area),
    and reorganizes the data by image name with adjusted bounding boxes. The bounding box shrinkage helps refine
    component boundaries for more accurate UI element interaction and matching.
    
    Args:
        reslut_file_root (str): The root directory path containing JSON files with detection results.
        shrink (int): The number of pixels to shrink each bounding box from all sides. Defaults to 3.
    
    Returns:
        dict: A dictionary where keys are image names (str) and values are dictionaries containing:
            - 'bboxes' (list): A list of bounding boxes, each represented as [column_min, row_min, column_max, row_max]
              with shrink applied to all coordinates.
            - 'categories' (list): A list of component categories (str) corresponding to each bounding box.
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
                compos_reform[img_name] = {'bboxes': [
                    [compo['column_min'] + shrink, compo['row_min'] + shrink, compo['column_max'] - shrink,
                     compo['row_max'] - shrink]],
                                           'categories': [compo['category']]}
            else:
                compos_reform[img_name]['bboxes'].append(
                    [compo['column_min'] + shrink, compo['row_min'] + shrink, compo['column_max'] - shrink,
                     compo['row_max'] - shrink])
                compos_reform[img_name]['categories'].append(compo['category'])
    return compos_reform


def load_ground_truth_json(gt_file):
    """
    Loads ground truth annotations from a JSON file and organizes them by image.
    
    This method reads a JSON file containing annotated UI components and their spatial
    information, then organizes the annotations into a dictionary keyed by image name.
    This structured organization enables efficient lookup of component locations and
    properties during UI analysis and element matching tasks.
    
    Args:
        gt_file: Path to the JSON file containing ground truth annotations in COCO format.
    
    Returns:
        A dictionary mapping image names to their annotations. Each image name maps to
        a dictionary containing:
        - 'bboxes': List of bounding boxes in [col_min, row_min, col_max, row_max] format
        - 'categories': List of category IDs corresponding to each bounding box
        - 'size': Tuple of (image height, image width)
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
    Evaluates detection results against ground truth annotations to measure UI component detection accuracy.
    
    This method compares detected UI components with ground truth annotations across multiple images,
    calculating true positives, false positives, and false negatives to assess detection performance.
    By matching detected bounding boxes against ground truth using intersection-over-union (IoU) metrics,
    the method quantifies how accurately the detection system identifies UI elements. Results are
    categorized by component size to understand detection performance across different UI element scales,
    which is essential for optimizing the detection pipeline used in interactive task automation workflows.
    The method supports filtering by component type (text vs non-text) to evaluate detection performance
    on specific UI element categories.
    
    Args:
        detection (dict): Dictionary mapping image IDs to detected components. Each entry contains:
            - 'bboxes': List of bounding boxes in format [col_min, row_min, col_max, row_max]
            - 'categories': List of component category labels corresponding to each bbox
        ground_truth (dict): Dictionary mapping image IDs to ground truth components with structure:
            - 'bboxes': List of ground truth bounding boxes
            - 'categories': List of ground truth category labels
            - 'size': Original image height used for coordinate normalization
        img_root (str): Path to the root directory containing image files for visualization.
        show (bool, optional): If True, displays images with detected (blue) and ground truth (red)
            bounding boxes overlaid. Defaults to True.
        no_text (bool, optional): If True, excludes text components (TextView) from evaluation.
            Defaults to False.
        only_text (bool, optional): If True, includes only text components (TextView) in evaluation.
            Defaults to False.
    
    Returns:
        None. Prints evaluation metrics at regular intervals (every 200 images) and upon completion,
        including true positives, false positives, false negatives, precision, recall, and F1-score
        for three size categories: small (<64 pixels), medium (64-128 pixels), and large (>128 pixels).
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
        size = -1
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
                if (gt_bbox[2] - gt_bbox[0]) < 64:
                    size = 0
                elif 64 < (gt_bbox[2] - gt_bbox[0]) < 128:
                    size = 1
                elif (gt_bbox[2] - gt_bbox[0]) > 128:
                    size = 2
                matched[i] = 0
                return True, size
        return False, size

    amount = len(detection)
    TP, FP, FN = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i, image_id in enumerate(detection):
        img = cv2.imread(pjoin(img_root, image_id + '.jpg'))
        d_compos = detection[image_id]
        if image_id not in ground_truth:
            continue
        gt_compos = ground_truth[image_id]

        org_height = gt_compos['size'][0]

        d_compos = compo_filter(d_compos, 'det')
        gt_compos = compo_filter(gt_compos, 'gt')

        d_compos['bboxes'] = resize_label(d_compos['bboxes'], 800, 1024)
        gt_compos['bboxes'] = resize_label(gt_compos['bboxes'], org_height, 1024)
        matched = np.ones(len(gt_compos['bboxes']), dtype=int)
        for d_bbox in d_compos['bboxes']:
            m, size = match(img, d_bbox, gt_compos['bboxes'], matched)
            if m:
                TP[size] += 1
            else:
                h = d_bbox[2] - d_bbox[0]
                if h < 64:
                    size = 0
                elif 64 < h < 128:
                    size = 1
                elif h > 128:
                    size = 2
                FP[size] += 1

        for i in range(len(matched)):
            if matched[i] == 1:
                gt_bboxes = gt_compos['bboxes']
                h = gt_bboxes[i][2] - gt_bboxes[i][0]
                if h < 64:
                    size = 0
                elif 64 < h < 128:
                    size = 1
                elif h > 128:
                    size = 2
                FN[size] += 1

        if show:
            print(image_id + '.jpg')
            # cv2.imshow('org', cv2.resize(img, (500, 1000)))
            broad = draw_bounding_box(img, d_compos['bboxes'], color=(255, 0, 0), line=3)
            draw_bounding_box(broad, gt_compos['bboxes'], color=(0, 0, 255), show=True, line=2)

        if i % 200 == 0:
            precision = [round(TP[i] / (TP[i] + FP[i]),3) for i in range(len(TP))]
            recall = [round(TP[i] / (TP[i] + FN[i]),3) for i in range(len(TP))]
            f1 = [round(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]), 3) for i in range(3)]
            print(
                '[%d/%d] TP:%s, FP:%s, FN:%s, Precesion:%s, Recall:%s, F1:%s' % (
                i, amount, str(TP), str(FP), str(FN), str(precision), str(recall), str(f1)))

    precision = [round(TP[i] / (TP[i] + FP[i]),3) for i in range(len(TP))]
    recall = [round(TP[i] / (TP[i] + FN[i]),3) for i in range(len(TP))]
    f1 = [round(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]), 3) for i in range(3)]
    print(
        '[%d/%d] TP:%s, FP:%s, FN:%s, Precesion:%s, Recall:%s, F1:%s' % (
            i, amount, str(TP), str(FP), str(FN), str(precision), str(recall), str(f1)))
    # print("Average precision:%.4f; Average recall:%.3f" % (sum(pres)/len(pres), sum(recalls)/len(recalls)))


no_text = False
only_text = False
detect = load_detect_result_json('E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_v3\\merge')
gt = load_ground_truth_json('E:\\Mulong\\Datasets\\rico\\instances_test.json')
eval(detect, gt, 'E:\\Mulong\\Datasets\\rico\\combined', show=False, no_text=no_text, only_text=only_text)