import cv2
import numpy as np

import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.Component import Component
import detect_compo.lib_ip.Component as Compo
from UIED_config.CONFIG_UIED import Config
C = Config()


def merge_intersected_corner(compos, org, is_merge_contained_ele, max_gap=(0, 0), max_ele_height=25):
    '''
    Recursively merge UI components that intersect or are contained within each other to consolidate 
    overlapping or adjacent elements into unified components. This process helps identify and group 
    related UI elements that should be treated as single interactive units.
    
    Args:
        compos (list): List of component objects to be merged
        org: Original image/screenshot for coordinate reference and shape information
        is_merge_contained_ele (bool): If True, merge components that are nested within other components
        max_gap (tuple): Maximum (horizontal_distance, vertical_distance) threshold for merging 
                         components into aligned rows or columns. Defaults to (0, 0)
        max_ele_height (int): Height threshold in pixels; components exceeding this are classified 
                              as text elements. Defaults to 25
    
    Returns:
        list: Merged list of components with intersecting or contained elements consolidated into 
              single components. Returns original list if no merging occurred, otherwise returns 
              recursively merged result
    '''
    changed = False
    new_compos = []
    Compo.compos_update(compos, org.shape)
    for i in range(len(compos)):
        merged = False
        cur_compo = compos[i]
        for j in range(len(new_compos)):
            relation = cur_compo.compo_relation(new_compos[j], max_gap)
            # print(relation)
            # draw.draw_bounding_box(org, [cur_compo, new_compos[j]], name='b-merge', show=True)
            # merge compo[i] to compo[j] if
            # 1. compo[j] contains compo[i]
            # 2. compo[j] intersects with compo[i] with certain iou
            # 3. is_merge_contained_ele and compo[j] is contained in compo[i]
            if relation == 1 or \
                    relation == 2 or \
                    (is_merge_contained_ele and relation == -1):
                # (relation == 2 and new_compos[j].height < max_ele_height and cur_compo.height < max_ele_height) or\

                new_compos[j].compo_merge(cur_compo)
                cur_compo = new_compos[j]
                # draw.draw_bounding_box(org, [new_compos[j]], name='a-merge', show=True)
                merged = True
                changed = True
                # break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_intersected_corner(new_compos, org, is_merge_contained_ele, max_gap, max_ele_height)


def merge_intersected_compos(compos):
    """
    Merges intersected components by iteratively combining overlapping components.
    
    This method processes a list of components and consolidates any that overlap,
    ensuring that detected UI elements are properly unified into distinct, non-overlapping
    regions. It repeatedly iterates through the components, checking each pair for
    intersection and merging them when detected, until no further merges are possible.
    This is essential for maintaining accurate UI element representation and preventing
    duplicate or fragmented component detection.
    
    Args:
        compos: A list of component objects to be merged. Each component should
            have compo_relation and compo_merge methods for intersection detection
            and merging operations.
    
    Returns:
        A list of merged components with all intersections resolved. Components
        that intersected with others have been combined, and the final list
        contains no overlapping components.
    """
    changed = True
    while changed:
        changed = False
        temp_set = []
        for compo_a in compos:
            merged = False
            for compo_b in temp_set:
                if compo_a.compo_relation(compo_b) == 2:
                    compo_b.compo_merge(compo_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(compo_a)
        compos = temp_set.copy()
    return compos


def rm_contained_compos_not_in_block(compos):
    '''
    Filter out redundant UI components that are spatially contained within other components.
    
    Removes components that are completely enclosed by other components, except for Block-type
    components which are preserved to maintain structural hierarchy. This prevents duplicate
    detection of nested UI elements and ensures only the most relevant components are retained
    for interaction and analysis.
    
    Args:
        compos (list): A list of component objects, each with a 'category' attribute and
                       a 'compo_relation' method that returns spatial relationship information.
    
    Returns:
        list: A filtered list of components with contained non-Block components removed.
              Components that contain others or are not contained by any other component
              are preserved.
    '''
    marked = np.full(len(compos), False)
    for i in range(len(compos) - 1):
        for j in range(i + 1, len(compos)):
            relation = compos[i].compo_relation(compos[j])
            if relation == -1 and compos[j].category != 'Block':
                marked[i] = True
            if relation == 1 and compos[i].category != 'Block':
                marked[j] = True
    new_compos = []
    for i in range(len(marked)):
        if not marked[i]:
            new_compos.append(compos[i])
    return new_compos


def merge_text(compos, org_shape, max_word_gad=4, max_word_height=20):
    """
    Recursively merges adjacent text components that are on the same line to consolidate fragmented UI elements into cohesive text regions.
    
    This method identifies text components that are horizontally aligned and close together, then merges them into single components. The process repeats until no more merges are possible. This is essential for accurately reconstructing text content from detected UI elements, ensuring that words or phrases split across multiple detection regions are properly unified for downstream processing and interaction.
    
    Args:
        compos: A list of component objects to be merged. Each component has methods
            put_bbox() to get bounding box coordinates (col_min, row_min, col_max, row_max)
            and compo_merge() to merge with another component.
        org_shape: The original shape of the image or canvas, used to provide context
            for the components. Typically a tuple containing (height, width, ...).
        max_word_gad: The maximum horizontal gap (in pixels) allowed between two
            components for them to be considered part of the same text line.
            Default is 4 pixels.
        max_word_height: The maximum height (in pixels) for a component to be
            considered as text. Components taller than this are not merged.
            Default is 20 pixels.
    
    Returns:
        A list of merged component objects. If no merges were performed, returns
        the original compos list. Otherwise, returns a recursively merged list
        where all eligible text components on the same line have been combined.
    """
    def is_text_line(compo_a, compo_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = compo_a.put_bbox()
        (col_min_b, row_min_b, col_max_b, row_max_b) = compo_b.put_bbox()

        col_min_s = max(col_min_a, col_min_b)
        col_max_s = min(col_max_a, col_max_b)
        row_min_s = max(row_min_a, row_min_b)
        row_max_s = min(row_max_a, row_max_b)

        # on the same line
        # if abs(row_min_a - row_min_b) < max_word_gad and abs(row_max_a - row_max_b) < max_word_gad:
        if row_min_s < row_max_s:
            # close distance
            if col_min_s < col_max_s or \
                    (0 < col_min_b - col_max_a < max_word_gad) or (0 < col_min_a - col_max_b < max_word_gad):
                return True
        return False

    changed = False
    new_compos = []
    row, col = org_shape[:2]
    for i in range(len(compos)):
        merged = False
        height = compos[i].height
        # ignore non-text
        # if height / row > max_word_height_ratio\
        #         or compos[i].category != 'Text':
        if height > max_word_height:
            new_compos.append(compos[i])
            continue
        for j in range(len(new_compos)):
            # if compos[j].category != 'Text':
            #     continue
            if is_text_line(compos[i], new_compos[j]):
                new_compos[j].compo_merge(compos[i])
                merged = True
                changed = True
                break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_text(new_compos, org_shape)


def rm_top_or_bottom_corners(components, org_shape, top_bottom_height=C.THRESHOLD_TOP_BOTTOM_BAR):
    """
    Filters out components located in the top or bottom corners of an image to isolate
    the main content area for UI element detection and analysis.
    
    This method removes components that fall within specified threshold regions at the
    top and bottom of the image, typically used to exclude header and footer bars that
    are not part of the interactive content. This filtering is essential for accurately
    identifying and locating actionable UI components within the primary content region.
    
    Args:
        components: A list of component objects to filter, where each component has
            a put_bbox() method that returns bounding box coordinates as
            (column_min, row_min, column_max, row_max).
        org_shape: A tuple or array containing the original image dimensions, where
            the first two elements are height and width respectively.
        top_bottom_height: A tuple of two float values representing the threshold
            ratios for the top and bottom regions. The first value defines the
            bottom boundary of the top region (as a fraction of image height),
            and the second value defines the top boundary of the bottom region
            (as a fraction of image height). Defaults to C.THRESHOLD_TOP_BOTTOM_BAR.
    
    Returns:
        A list of component objects that are not located in the top or bottom
        corner regions as defined by the threshold parameters, representing the
        main content area components suitable for UI analysis and interaction.
    """
    new_compos = []
    height, width = org_shape[:2]
    for compo in components:
        (column_min, row_min, column_max, row_max) = compo.put_bbox()
        # remove big ones
        # if (row_max - row_min) / height > 0.65 and (column_max - column_min) / width > 0.8:
        #     continue
        if not (row_max < height * top_bottom_height[0] or row_min > height * top_bottom_height[1]):
            new_compos.append(compo)
    return new_compos


def rm_line_v_h(binary, show=False, max_line_thickness=C.THRESHOLD_LINE_THICKNESS):
    """
    Removes vertical and horizontal lines from a binary image to isolate UI content.
    
    This method detects and extracts continuous vertical and horizontal lines from
    a binary image, then removes them by subtracting the detected line areas from
    the original image. This preprocessing step helps isolate meaningful UI elements
    by eliminating structural lines that may interfere with element detection and
    analysis. Lines are identified based on continuous pixel regions that span at
    least 60% of the image dimension (width for horizontal lines, height for vertical
    lines) and have a thickness below the specified threshold.
    
    Args:
        binary (numpy.ndarray): A binary image from which lines will be removed.
        show (bool, optional): A boolean flag indicating whether to display intermediate
            results using OpenCV image windows. Defaults to False.
        max_line_thickness (int, optional): The maximum thickness (in pixels) for a
            region to be considered a line. Regions thicker than this value are not
            removed. Defaults to C.THRESHOLD_LINE_THICKNESS.
    
    Returns:
        numpy.ndarray: The modified binary image with detected vertical and horizontal
            lines removed (subtracted from the original image).
    """
    def check_continuous_line(line, edge):
        continuous_length = 0
        line_start = -1
        for j, p in enumerate(line):
            if p > 0:
                if line_start == -1:
                    line_start = j
                continuous_length += 1
            elif continuous_length > 0:
                if continuous_length / edge > 0.6:
                    return [line_start, j]
                continuous_length = 0
                line_start = -1

        if continuous_length / edge > 0.6:
            return [line_start, len(line)]
        else:
            return None

    def extract_line_area(line, start_idx, flag='v'):
        for e, l in enumerate(line):
            if flag == 'v':
                map_line[start_idx + e, l[0]:l[1]] = binary[start_idx + e, l[0]:l[1]]

    map_line = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.imshow('binary', binary)

    width = binary.shape[1]
    start_row = -1
    line_area = []
    for i, row in enumerate(binary):
        line_v = check_continuous_line(row, width)
        if line_v is not None:
            # new line
            if start_row == -1:
                start_row = i
                line_area = []
            line_area.append(line_v)
        else:
            # checking line
            if start_row != -1:
                if i - start_row < max_line_thickness:
                    # binary[start_row: i] = 0
                    # map_line[start_row: i] = binary[start_row: i]
                    print(line_area, start_row, i)
                    extract_line_area(line_area, start_row)
                start_row = -1

    height = binary.shape[0]
    start_col = -1
    for i in range(width):
        col = binary[:, i]
        line_h = check_continuous_line(col, height)
        if line_h is not None:
            # new line
            if start_col == -1:
                start_col = i
        else:
            # checking line
            if start_col != -1:
                if i - start_col < max_line_thickness:
                    # binary[:, start_col: i] = 0
                    map_line[:, start_col: i] = binary[:, start_col: i]
                start_col = -1

    binary -= map_line

    if show:
        cv2.imshow('no-line', binary)
        cv2.imshow('lines', map_line)
        cv2.waitKey()


def rm_line(binary,
            max_line_thickness=C.THRESHOLD_LINE_THICKNESS,
            min_line_length_ratio=C.THRESHOLD_LINE_MIN_LENGTH,
            show=False, wait_key=0):
    """
    Removes horizontal lines from a binary image to clean up document structure for UI element detection.
    
    This method identifies and removes horizontal lines from a binary image by analyzing
    consecutive rows for valid line patterns. A line is considered valid if it spans a
    significant portion of the image width with minimal gaps. Lines are removed if they
    are thin enough and either reach the image boundaries or have sufficient gaps between
    them. This preprocessing step helps isolate UI components by eliminating structural
    lines that may interfere with element detection and analysis.
    
    Args:
        binary: A binary image (numpy array) from which lines will be removed.
        max_line_thickness: Maximum thickness in pixels for a valid line to be removed.
            Defaults to C.THRESHOLD_LINE_THICKNESS.
        min_line_length_ratio: Minimum ratio of line length to image width for validity.
            Defaults to C.THRESHOLD_LINE_MIN_LENGTH.
        show: Whether to display the resulting image with lines removed. Defaults to False.
        wait_key: Key wait time in milliseconds for the displayed image. If 0, the window
            is destroyed immediately after display. Defaults to 0.
    
    Returns:
        None. The method modifies the binary image in-place by setting identified line
        regions to 0 (black).
    """
    def is_valid_line(line):
        line_length = 0
        line_gap = 0
        for j in line:
            if j > 0:
                if line_gap > 5:
                    return False
                line_length += 1
                line_gap = 0
            elif line_length > 0:
                line_gap += 1
        if line_length / width > 0.95:
            return True
        return False

    height, width = binary.shape[:2]
    board = np.zeros(binary.shape[:2], dtype=np.uint8)

    start_row, end_row = -1, -1
    check_line = False
    check_gap = False
    for i, row in enumerate(binary):
        # line_ratio = (sum(row) / 255) / width
        # if line_ratio > 0.9:
        if is_valid_line(row):
            # new start: if it is checking a new line, mark this row as start
            if not check_line:
                start_row = i
                check_line = True
        else:
            # end the line
            if check_line:
                # thin enough to be a line, then start checking gap
                if i - start_row < max_line_thickness:
                    end_row = i
                    check_gap = True
                else:
                    start_row, end_row = -1, -1
                check_line = False
        # check gap
        if check_gap and i - end_row > max_line_thickness:
            binary[start_row: end_row] = 0
            start_row, end_row = -1, -1
            check_line = False
            check_gap = False

    if (check_line and (height - start_row) < max_line_thickness) or check_gap:
        binary[start_row: end_row] = 0

    if show:
        cv2.imshow('no-line binary', binary)
        if wait_key is not None:
            cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow('no-line binary')


def rm_noise_compos(compos):
    """
    Removes noise components from a composition list to ensure only meaningful UI elements are retained for analysis.
    
    This method filters out all components with a category of 'Noise' from the
    provided composition list, returning a new list containing only non-noise
    components. This is essential for maintaining clean detection results when
    analyzing UI elements, as noise components can interfere with accurate element
    matching and interaction guidance.
    
    Args:
        compos: A list of composition objects to filter.
    
    Returns:
        list: A new list containing all components from the input list except those
        with a category of 'Noise'.
    """
    compos_new = []
    for compo in compos:
        if compo.category == 'Noise':
            continue
        compos_new.append(compo)
    return compos_new


def rm_noise_in_large_img(compos, org,
                      max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    """
    Filters out noise components that are contained within larger image-category components.
    
    This method identifies and removes UI elements that are nested within image components,
    keeping only the top-level components that are not contained by any image-category elements.
    This is essential for maintaining a clean component hierarchy when analyzing complex UI layouts,
    ensuring that only meaningful, non-redundant UI elements are retained for further processing.
    
    Args:
        compos: A list of component objects detected in the image, each with
            category and contain attributes.
        org: The original image as a numpy array from which components were detected.
        max_compo_scale: The maximum scale threshold for component filtering
            (default uses C.THRESHOLD_COMPO_MAX_SCALE constant).
    
    Returns:
        A filtered list of component objects with noise components removed,
        containing only components that are not contained within larger
        image-category components.
    """
    row, column = org.shape[:2]
    remain = np.full(len(compos), True)
    new_compos = []
    for compo in compos:
        if compo.category == 'Image':
            for i in compo.contain:
                remain[i] = False
    for i in range(len(remain)):
        if remain[i]:
            new_compos.append(compos[i])
    return new_compos


def detect_compos_in_img(compos, binary, org, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE, show=False):
    """
    Detects and extracts rectangular components nested within image regions to support UI element analysis.
    
    This method identifies structural elements within image-type components by performing binary image
    analysis on their regions. For each image component, it clips the binary representation to the component's
    bounding box, inverts the binary values to highlight internal structures, and applies component detection
    to extract rectangular sub-elements. This enables the system to discover and catalog UI components that
    may be embedded within larger image regions, supporting comprehensive UI element mapping for task automation.
    
    Args:
        compos: List of component objects to process, filtered for 'Image' category components.
        binary: Binary image array used for component detection within image regions.
        org: Original image array (used for reference, though not directly utilized in current implementation).
        max_compo_scale: Maximum scale threshold for component filtering (default from C.THRESHOLD_COMPO_MAX_SCALE).
        show: Boolean flag to control visualization of intermediate processing steps (default False).
    
    Returns:
        None. The method modifies the input compos list in-place by appending newly detected
        rectangular components that meet the filtering criteria (area ratio < 0.8 relative to
        parent component, height > 20 pixels, width > 20 pixels).
    """
    compos_new = []
    row, column = binary.shape[:2]
    for compo in compos:
        if compo.category == 'Image':
            compo.compo_update_bbox_area()
            # org_clip = compo.compo_clipping(org)
            # bin_clip = pre.binarization(org_clip, show=show)
            bin_clip = compo.compo_clipping(binary)
            bin_clip = pre.reverse_binary(bin_clip, show=show)

            compos_rec, compos_nonrec = component_detection(bin_clip, test=False, step_h=10, step_v=10, rec_detect=True)
            for compo_rec in compos_rec:
                compo_rec.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
                if compo_rec.bbox_area / compo.bbox_area < 0.8 and compo_rec.bbox.height > 20 and compo_rec.bbox.width > 20:
                    compos_new.append(compo_rec)
                    # draw.draw_bounding_box(org, [compo_rec], show=True)

            # compos_inner = component_detection(bin_clip, rec_detect=False)
            # for compo_inner in compos_inner:
            #     compo_inner.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
            #     draw.draw_bounding_box(org, [compo_inner], show=True)
            #     if compo_inner.bbox_area / compo.bbox_area < 0.8:
            #         compos_new.append(compo_inner)
    compos += compos_new


def compo_filter(compos, min_area, img_shape):
    """
    Filters detected UI components based on size and shape constraints to identify valid interactive elements.
    
    This method removes components that are too small, too large, or have invalid aspect ratios,
    ensuring only meaningful UI elements are retained for interaction guidance. Components are validated
    against minimum area requirements, maximum height relative to image dimensions, and aspect ratio
    constraints to eliminate noise, oversized elements, and extremely elongated or thin artifacts that
    are unlikely to represent actionable UI components.
    
    Args:
        compos: A list of component objects to be filtered, where each component has
            properties like area, height, and width representing detected UI elements.
        min_area: The minimum area threshold in pixels; components with area below this value
            are excluded from the result.
        img_shape: A tuple representing the image dimensions (height, width), where the first element
            is the image height used to calculate the maximum allowable component height (80% of image height).
    
    Returns:
        A list of filtered component objects that satisfy all filtering criteria: minimum
        area requirement, maximum height constraint (80% of image height), and acceptable aspect ratios
        (width/height ratio ≤ 50, height/width ratio ≤ 40, with stricter constraints for very small components).
    """
    max_height = img_shape[0] * 0.8
    compos_new = []
    for compo in compos:
        if compo.area < min_area:
            continue
        if compo.height > max_height:
            continue
        ratio_h = compo.width / compo.height
        ratio_w = compo.height / compo.width
        if ratio_h > 50 or ratio_w > 40 or \
                (min(compo.height, compo.width) < 8 and max(ratio_h, ratio_w) > 10):
            continue
        compos_new.append(compo)
    return compos_new


def is_block(clip, thread=0.15):
    '''
    Detect if a rectangular region represents a block element (wireframe container) by analyzing the blankness of its inner borders.
    
    A block is a rectangle border that encloses a group of UI components, forming a wireframe-like structure.
    This method validates whether a region qualifies as a block by scanning the inner edges of all four borders
    and checking if they contain sufficient blank space, indicating an empty container rather than a filled element.
    
    Args:
        clip: A 2D numpy array representing the image region to analyze (typically a binary or grayscale image).
        thread (float): The threshold ratio (0.0-1.0) for determining if a border line is blank. 
                       A line is considered blank if its non-zero pixel sum divided by 255 exceeds this threshold 
                       multiplied by the line's length. Default is 0.15.
    
    Returns:
        bool: True if the region is identified as a block (has blank inner borders on all sides), 
              False otherwise.
    '''
    side = 4  # scan 4 lines inner forward each border
    # top border - scan top down
    blank_count = 0
    for i in range(1, 5):
        if sum(clip[side + i]) / 255 > thread * clip.shape[1]:
            blank_count += 1
    if blank_count > 2: return False
    # left border - scan left to right
    blank_count = 0
    for i in range(1, 5):
        if sum(clip[:, side + i]) / 255 > thread * clip.shape[0]:
            blank_count += 1
    if blank_count > 2: return False

    side = -4
    # bottom border - scan bottom up
    blank_count = 0
    for i in range(-1, -5, -1):
        if sum(clip[side + i]) / 255 > thread * clip.shape[1]:
            blank_count += 1
    if blank_count > 2: return False
    # right border - scan right to left
    blank_count = 0
    for i in range(-1, -5, -1):
        if sum(clip[:, side + i]) / 255 > thread * clip.shape[0]:
            blank_count += 1
    if blank_count > 2: return False
    return True


def compo_block_recognition(binary, compos, block_side_length=0.15):
    """
    Recognizes and categorizes components as blocks based on size and content analysis.
    
    This method identifies substantial UI components that represent significant content areas
    by analyzing their dimensions relative to the screen and validating their content structure.
    Components meeting both size and content criteria are marked as 'Block' elements, enabling
    the system to distinguish major content regions from smaller UI elements during screen analysis.
    
    Args:
        binary: A binary image represented as a 2D array from which component dimensions
            are derived.
        compos: A list of component objects to be analyzed and potentially categorized.
        block_side_length: The minimum relative size threshold (as a fraction of image
            dimensions) that a component must exceed in both height and width to be
            considered for block classification. Defaults to 0.15.
    
    Returns:
        None. The method modifies the category attribute of component objects in-place,
        setting the category to 'Block' for components that meet the size and content criteria.
    """
    height, width = binary.shape
    for compo in compos:
        if compo.height / height > block_side_length and compo.width / width > block_side_length:
            clip = compo.compo_clipping(binary)
            if is_block(clip):
                compo.category = 'Block'


# take the binary image as input
# calculate the connected regions -> get the bounding boundaries of them -> check if those regions are rectangles
# return all boundaries and boundaries of rectangles
def component_detection(binary, min_obj_area,
                        line_thickness=C.THRESHOLD_LINE_THICKNESS,
                        min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                        max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                        step_h = 5, step_v = 2,
                        rec_detect=False, show=False, test=False):
    """
    Detects and extracts connected components from a binary image to identify UI elements and interactive regions.
    
    This method performs flood-fill based connected component analysis on a binary image to locate distinct
    visual elements. Each detected component is validated against size and shape criteria to filter out noise
    and irrelevant artifacts, ensuring only meaningful UI elements are extracted for further processing.
    
    Args:
        binary (np.ndarray): Binary image from pre-processing where foreground pixels are 255.
        min_obj_area (int): Minimum pixel area threshold; components smaller than this are discarded.
        line_thickness (int, optional): Minimum thickness threshold to filter out line-like objects.
            Defaults to C.THRESHOLD_LINE_THICKNESS.
        min_rec_evenness (float, optional): Minimum evenness ratio for rectangular shape validation.
            Defaults to C.THRESHOLD_REC_MIN_EVENNESS.
        max_dent_ratio (float, optional): Maximum dent ratio for rectangular shape validation.
            Defaults to C.THRESHOLD_REC_MAX_DENT_RATIO.
        step_h (int, optional): Vertical step size for scanning pixels. Defaults to 5.
        step_v (int, optional): Horizontal step size for scanning pixels. Defaults to 2.
        rec_detect (bool, optional): If True, classifies components as rectangular or non-rectangular.
            Defaults to False.
        show (bool, optional): If True, displays detected components during processing. Defaults to False.
        test (bool, optional): If True, enables debug output. Defaults to False.
    
    Returns:
        list or tuple: If rec_detect is False, returns list of Component objects representing all detected
            elements. If rec_detect is True, returns tuple of (compos_rec, compos_nonrec) where compos_rec
            contains rectangular components and compos_nonrec contains non-rectangular components.
    """
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    compos_all = []
    compos_rec = []
    compos_nonrec = []
    row, column = binary.shape[0], binary.shape[1]
    for i in range(0, row, step_h):
        for j in range(i % 2, column, step_v):
            if binary[i, j] == 255 and mask[i, j] == 0:
                # get connected area
                # region = util.boundary_bfs_connected_area(binary, i, j, mask)

                mask_copy = mask.copy()
                ff = cv2.floodFill(binary, mask, (j, i), None, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                if ff[0] < min_obj_area: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                # filter out some compos
                component = Component(region, binary.shape)
                # calculate the boundary of the connected area
                # ignore small area
                if component.width <= 3 or component.height <= 3:
                    continue
                # check if it is line by checking the length of edges
                # if component.compo_is_line(line_thickness):
                #     continue

                if test:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary([component], binary.shape, show=True)

                compos_all.append(component)

                if rec_detect:
                    # rectangle check
                    if component.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                        component.rect_ = True
                        compos_rec.append(component)
                    else:
                        component.rect_ = False
                        compos_nonrec.append(component)

                if show:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary(compos_all, binary.shape, show=True)

    # draw.draw_boundary(compos_all, binary.shape, show=True)
    if rec_detect:
        return compos_rec, compos_nonrec
    else:
        return compos_all


def nested_components_detection(grey, org, grad_thresh,
                   show=False, write_path=None,
                   step_h=10, step_v=10,
                   line_thickness=C.THRESHOLD_LINE_THICKNESS,
                   min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                   max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO):
    """
    Detects rectangular UI components in an image by identifying connected regions of similar intensity.
    
    This method performs flood-fill based region detection to locate distinct UI blocks and containers
    within a layout. It filters detected regions to retain only valid rectangular components that meet
    geometric and structural criteria, excluding noise, lines, and overly large background areas.
    
    Args:
        grey (np.ndarray): Grayscale image array for region detection
        org (np.ndarray): Original image array for reference
        grad_thresh (int): Gradient threshold for flood-fill algorithm
        show (bool, optional): Whether to display intermediate detection results. Defaults to False
        write_path (str, optional): File path to save detection visualization. Defaults to None
        step_h (int, optional): Vertical step size for region scanning. Defaults to 10
        step_v (int, optional): Horizontal step size for region scanning. Defaults to 10
        line_thickness (int, optional): Maximum thickness to classify a region as a line. Defaults to C.THRESHOLD_LINE_THICKNESS
        min_rec_evenness (float, optional): Minimum evenness ratio for rectangle validation. Defaults to C.THRESHOLD_REC_MIN_EVENNESS
        max_dent_ratio (float, optional): Maximum dent ratio for rectangle validation. Defaults to C.THRESHOLD_REC_MAX_DENT_RATIO
    
    Returns:
        list[Component]: List of detected rectangular components, each containing:
            - Bounding box coordinates (top_left, bottom_right)
            - top_left: (column_min, row_min)
            - bottom_right: (column_max, row_max)
            - Component properties (area, height, width, redundancy flag)
    """
    compos = []
    mask = np.zeros((grey.shape[0]+2, grey.shape[1]+2), dtype=np.uint8)
    broad = np.zeros((grey.shape[0], grey.shape[1], 3), dtype=np.uint8)
    broad_all = broad.copy()

    row, column = grey.shape[0], grey.shape[1]
    for x in range(0, row, step_h):
        for y in range(0, column, step_v):
            if mask[x, y] == 0:
                # region = flood_fill_bfs(grey, x, y, mask)

                # flood fill algorithm to get background (layout block)
                mask_copy = mask.copy()
                ff = cv2.floodFill(grey, mask, (y, x), None, grad_thresh, grad_thresh, cv2.FLOODFILL_MASK_ONLY)
                # ignore small regions
                if ff[0] < 500: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                compo = Component(region, grey.shape)
                # draw.draw_region(region, broad_all)
                # if block.height < 40 and block.width < 40:
                #     continue
                if compo.height < 30:
                    continue

                # print(block.area / (row * column))
                if compo.area / (row * column) > 0.9:
                    continue
                elif compo.area / (row * column) > 0.7:
                    compo.redundant = True

                # get the boundary of this region
                # ignore lines
                if compo.compo_is_line(line_thickness):
                    continue
                # ignore non-rectangle as blocks must be rectangular
                if not compo.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                    continue
                # if block.height/row < min_block_height_ratio:
                #     continue
                compos.append(compo)
                # draw.draw_region(region, broad)
    if show:
        cv2.imshow('flood-fill all', broad_all)
        cv2.imshow('block', broad)
        cv2.waitKey()
    if write_path is not None:
        cv2.imwrite(write_path, broad)
    return compos
