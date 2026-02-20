from tqdm import tqdm
import json
import cv2
from os.path import join as pjoin

from UIED_config.CONFIG_UIED import Config
C = Config()


def draw_bounding_box_class(org, components, color=C.COLOR, line=2, show=False, write_path=None):
    """
    Visualize detected UI components with their classification labels on the original image to provide interactive guidance for task automation.
    
    This method annotates the input image with bounding boxes and class labels for each detected component, enabling users to understand which UI elements have been identified and their corresponding types. This visual feedback is essential for verifying component detection accuracy and guiding users through the automated task workflow.
    
    Args:
        org: Original input image (numpy array in BGR format)
        components: Dictionary containing detected components with keys:
            - 'bboxes': List of bounding boxes in format [(column_min, row_min, column_max, row_max)]
                        where top_left: (column_min, row_min) and bottom_right: (column_max, row_max)
            - 'categories': List of category indices corresponding to each bbox
        color: Color mapping dictionary for different component classes (default: C.COLOR)
        line: Thickness of bounding box lines in pixels (default: 2)
        show: Whether to display the annotated image in a window (default: False)
        write_path: Optional file path to save the annotated image (default: None)
    
    Returns:
        Annotated image with drawn bounding boxes and class labels for all detected components
    """
    board = org.copy()
    bboxes = components['bboxes']
    categories = components['categories']
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        category = categories[i]
        board = cv2.rectangle(board, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[C.CLASS_MAP[str(category)]], line)
        board = cv2.putText(board, C.CLASS_MAP[str(category)], (bbox[0]+5, bbox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[C.CLASS_MAP[str(category)]], 2)
    if show:
        cv2.imshow('a', cv2.resize(board, (500, 1000)))
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def load_ground_truth_json(gt_file, no_text=True):
    """
    Loads ground truth annotations from a JSON file and organizes them by image to support UI element detection and analysis.
    
    This method reads a COCO-format JSON file containing image and annotation data, extracts bounding boxes and categories for each image, and returns a dictionary mapping image names to their corresponding UI components. This enables the system to match detected UI elements against known ground truth data for validation and analysis purposes.
    
    Args:
        gt_file: Path to the ground truth JSON file in COCO format containing
            'images' and 'annotations' keys.
        no_text: Flag to exclude text annotations (category_id == 14) from the
            results. Defaults to True.
    
    Returns:
        A dictionary where keys are image names (without extension) and values are
        dictionaries containing:
        - 'bboxes': List of bounding boxes in [col_min, row_min, col_max, row_max] format.
        - 'categories': List of category IDs corresponding to each bounding box.
        - 'size': Tuple of (height, width) for the image.
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
        if no_text and int(annot['category_id']) == 14:
            compos[img_name] = {'bboxes': [], 'categories': [], 'size': size}
            continue
        if img_name not in compos:
            compos[img_name] = {'bboxes': [cvt_bbox(annot['bbox'])], 'categories': [annot['category_id']], 'size':size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def view_gt_all(gt, img_root):
    """
    Visualizes ground truth component annotations across all images in a dataset.
        
    This method iterates through all images in the ground truth dictionary,
    loads each image from disk, and displays the associated bounding boxes
    with their class labels overlaid on the image. This visualization helps
    verify that UI components have been correctly detected and annotated,
    ensuring the accuracy of the component detection pipeline before further
    processing or model training.
        
    Args:
        gt (dict): A dictionary mapping image identifiers to their corresponding
            component/bounding box annotations. Each key is an image identifier
            and each value contains the bounding box data for components in that image.
        img_root (str): The root directory path where the image files are stored.
            Images are expected to be named as '{img_id}.jpg'.
        
    Returns:
        None. The method displays images with bounding boxes interactively but does not
        return any value.
    """
    for img_id in gt:
        compos = gt[img_id]
        img = cv2.imread(pjoin(img_root, img_id + '.jpg'))
        print(pjoin(img_root, img_id + '.jpg'))
        draw_bounding_box_class(img, compos, show=True)


def view_gt_single(gt, img_root, img_id):
    """
    Visualizes ground truth UI component annotations for a single image.
    
    This method retrieves the annotated UI components for a specified image,
    loads the image from disk, and displays the bounding boxes with their
    class labels overlaid on the image. This is useful for verifying that
    UI elements have been correctly identified and labeled during the annotation
    process.
    
    Args:
        gt: A dictionary mapping image IDs to their corresponding ground truth
            UI component annotations.
        img_root: The directory path where the image files are stored.
        img_id: The identifier of the image to visualize. Will be converted to
                a string if not already.
    
    Returns:
        None. The method displays the annotated image in a window but does not
        return any value.
    """
    img_id = str(img_id)
    compos = gt[img_id]
    img = cv2.imread(pjoin(img_root, img_id + '.jpg'))
    print(pjoin(img_root, img_id + '.jpg'))
    draw_bounding_box_class(img, compos, show=True)


gt = load_ground_truth_json('E:\\Mulong\\Datasets\\rico\\instances_test.json', no_text=False)
# view_gt_all(gt, 'E:\\Mulong\\Datasets\\rico\\combined')
view_gt_single(gt, 'E:\\Mulong\\Datasets\\rico\\combined', 670)
