import cv2
import numpy as np
from random import randint as rint
import time

import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_segment as seg
from detect_compo.lib_ip.Block import Block
from UIED_config.CONFIG_UIED import Config
C = Config()


def block_hierarchy(blocks):
    """
    Establishes the hierarchical relationships between blocks based on their compositional containment.
    
    This method builds a parent-child hierarchy by analyzing compositional relationships between
    all block pairs. For each pair of blocks, it determines whether one block contains the other
    by checking their compositional relation. When block i contains block j (relation == 1),
    block j is registered as a child of block i. Conversely, when block j contains block i
    (relation == -1), block i is registered as a child of block j. This hierarchical structure
    enables efficient navigation and organization of nested UI components.
    
    Args:
        blocks: A list of block objects, each having a compo_relation method to determine
            containment relationships with other blocks, and a children attribute (list) to
            store indices of child blocks.
    
    Returns:
        None
    """
    for i in range(len(blocks) - 1):
        for j in range(i + 1, len(blocks)):
            relation = blocks[i].compo_relation(blocks[j])
            if relation == -1:
                blocks[j].children.append(i)
            if relation == 1:
                blocks[i].children.append(j)
    return


def block_bin_erase_all_blk(binary, blocks, pad=0, show=False):
    '''
    Remove detected layout blocks from the binary map to isolate text and content regions.
    
    This method systematically erases all identified block regions from a binary image representation,
    which is essential for separating structural layout elements from the actual document content.
    By removing blocks, the method enables focused analysis of text and meaningful content areas
    without interference from detected layout structures.
    
    Args:
        binary (numpy.ndarray): Binary map of the original image where blocks will be erased.
        blocks (list): List of block objects with detected layout block boundaries.
        pad (int, optional): Padding value to expand the bounding boxes of blocks before erasing.
            Defaults to 0.
        show (bool, optional): If True, displays before and after visualization of the binary map.
            Defaults to False.
    
    Returns:
        numpy.ndarray: Modified binary map with all block regions erased, containing only
            non-block content.
    '''

    bin_org = binary.copy()
    for block in blocks:
        block.block_erase_from_bin(binary, pad)
    if show:
        cv2.imshow('before', bin_org)
        cv2.imshow('after', binary)
        cv2.waitKey()


def block_division(grey, org, grad_thresh,
                   show=False, write_path=None,
                   step_h=10, step_v=10,
                   line_thickness=C.THRESHOLD_LINE_THICKNESS,
                   min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                   max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                   min_block_height_ratio=C.THRESHOLD_BLOCK_MIN_HEIGHT):
    '''
    Divides an image into rectangular layout blocks by detecting connected regions of similar intensity.
    
    This method performs flood-fill based segmentation to identify distinct layout components (blocks)
    in a document or UI image. It filters detected regions to retain only valid rectangular blocks
    that represent meaningful layout elements, excluding noise, lines, and overly large background areas.
    
    Args:
        grey (np.ndarray): Grayscale image array of shape (height, width)
        org (np.ndarray): Original color image array
        grad_thresh (int): Gradient threshold for flood-fill algorithm
        show (bool, optional): Whether to display intermediate results. Defaults to False
        write_path (str, optional): Path to save the block detection result image. Defaults to None
        step_h (int, optional): Vertical step size for region scanning in pixels. Defaults to 10
        step_v (int, optional): Horizontal step size for region scanning in pixels. Defaults to 10
        line_thickness (int, optional): Maximum thickness to classify a region as a line. Defaults to C.THRESHOLD_LINE_THICKNESS
        min_rec_evenness (float, optional): Minimum evenness ratio for rectangle validation. Defaults to C.THRESHOLD_REC_MIN_EVENNESS
        max_dent_ratio (float, optional): Maximum dent ratio for rectangle validation. Defaults to C.THRESHOLD_REC_MAX_DENT_RATIO
        min_block_height_ratio (float, optional): Minimum height ratio relative to image height. Defaults to C.THRESHOLD_BLOCK_MIN_HEIGHT
    
    Returns:
        list[Block]: List of detected rectangular blocks, each containing region coordinates and properties
                     Block objects include geometric information (height, width, area) and validation flags
    '''
    blocks = []
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

                block = Block(region, grey.shape)
                # draw.draw_region(region, broad_all)
                # if block.height < 40 and block.width < 40:
                #     continue
                if block.height < 30:
                    continue

                # print(block.area / (row * column))
                if block.area / (row * column) > 0.9:
                    continue
                elif block.area / (row * column) > 0.7:
                    block.redundant = True

                # get the boundary of this region
                # ignore lines
                if block.compo_is_line(line_thickness):
                    continue
                # ignore non-rectangle as blocks must be rectangular
                if not block.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                    continue
                # if block.height/row < min_block_height_ratio:
                #     continue
                blocks.append(block)
                # draw.draw_region(region, broad)
    if show:
        cv2.imshow('flood-fill all', broad_all)
        cv2.imshow('block', broad)
        cv2.waitKey()
    if write_path is not None:
        cv2.imwrite(write_path, broad)
    return blocks
