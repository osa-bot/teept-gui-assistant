import cv2
from os.path import join as pjoin
import time
import numpy as np

from detect_compo.lib_ip.Component import Component
from UIED_config.CONFIG_UIED import Config
C = Config()


class Block(Component):
    """
    Represents a hierarchical block structure for organizing and managing UI components within an image.
    
        The Block class serves as a container for organizing rectangular regions in an image with support
        for hierarchical relationships (parent-child), UI component association, and binary image manipulation.
        It provides functionality to classify blocks based on their properties and remove them from binary images.
    
        Attributes:
            category: A string identifier set to 'Block' to classify this object type.
            parent: A reference to the parent block in the hierarchy, initialized to None.
            children: A list to store child blocks in the hierarchy, initialized as empty.
            uicompo_: A reference to an associated UI component, initialized to None.
            top_or_botm: A flag indicating whether the block is at the top or bottom, initialized to None.
            redundant: A boolean flag indicating whether the block is redundant, initialized to False.
    
        Methods:
            - __init__: Initializes a Block object with region and image shape information.
            - block_is_uicompo: Checks if the block is a UI component according to its relative size.
            - block_is_top_or_bottom_bar: Checks if the block is a top bar or bottom bar.
            - block_erase_from_bin: Erases a rectangular block from a binary image with optional padding.
    """

    def __init__(self, region, image_shape):
        """
        Initializes a Block object to represent a detected UI region within the screen layout hierarchy.
        
        This constructor creates a new Block instance by calling the parent class constructor 
        and initializing block-specific attributes that enable hierarchical organization of UI 
        elements and their association with detected UI components. Blocks serve as fundamental 
        units in the spatial decomposition of the screen, allowing the system to map detected 
        regions to actionable UI areas and maintain parent-child relationships for nested 
        component structures.
        
        Args:
            region: The region coordinates or bounds defining the block's location on the screen.
            image_shape: The shape or dimensions of the image containing the block.
        
        Returns:
            None
        
        Attributes:
            category: A string identifier set to 'Block' to classify this object type.
            parent: A reference to the parent block in the hierarchy, initialized to None.
            children: A list to store child blocks in the hierarchy, initialized as empty.
            uicompo_: A reference to an associated UI component detected within this region, 
                initialized to None.
            top_or_botm: A flag indicating whether the block is positioned at the top or bottom 
                of its parent, initialized to None.
            redundant: A boolean flag indicating whether the block is redundant or duplicate, 
                initialized to False.
        """
        super().__init__(region, image_shape)
        self.category = 'Block'
        self.parent = None
        self.children = []
        self.uicompo_ = None
        self.top_or_botm = None
        self.redundant = False

    def block_is_uicompo(self, image_shape, max_compo_scale):
        """
        Validate whether a block represents a valid UI component based on its dimensions relative to the image.
        
        This method filters out oversized blocks that exceed the expected scale of interactive UI components,
        ensuring that only appropriately-sized elements are considered as valid components for interaction.
        Blocks that are too large relative to the image dimensions are rejected as they likely represent
        layout containers or non-interactive regions rather than actionable UI elements.
        
        Args:
            image_shape (tuple): The shape of the image as (height, width, ...), where the first two elements
                                represent the row and column dimensions of the image.
            max_compo_scale (tuple): A tuple of (max_height_ratio, max_width_ratio) defining the maximum
                                    acceptable size of a UI component relative to the image dimensions.
        
        Returns:
            bool: True if the block is within acceptable size bounds and qualifies as a UI component,
                  False if the block exceeds the maximum scale thresholds.
        """
        row, column = image_shape[:2]
        # print(height, height / row, max_compo_scale[0], height / row > max_compo_scale[0])
        # draw.draw_bounding_box(org, [corner], show=True)
        # ignore atomic components
        if self.bbox.height / row > max_compo_scale[0] or self.bbox.width / column > max_compo_scale[1]:
            return False
        return True

    def block_is_top_or_bottom_bar(self, image_shape, top_bottom_height):
        """
        Identify whether the block represents a top or bottom navigation bar.
        
        This method detects UI bars that span the full width of the screen and are positioned
        at the top or bottom edges, which are common navigation or status bar components in GUI layouts.
        The detection uses proximity thresholds to identify blocks that extend nearly edge-to-edge
        horizontally and occupy minimal vertical space at screen boundaries.
        
        Args:
            image_shape (tuple): The shape of the image as (height, width, ...).
            top_bottom_height (tuple): A tuple of two float values representing the relative height
                thresholds for top bar (top_bottom_height[0]) and bottom bar (top_bottom_height[1])
                detection, expressed as fractions of total image height.
        
        Returns:
            bool: True if the block is identified as a top or bottom bar, False otherwise.
                When True, also sets the block's uicompo_ attribute to True to mark it as
                a UI component.
        """
        height, width = image_shape[:2]
        (column_min, row_min, column_max, row_max) = self.bbox.put_bbox()
        if column_min < 5 and row_min < 5 and \
                width - column_max < 5 and row_max < height * top_bottom_height[0]:
            self.uicompo_ = True
            return True
        if column_min < 5 and row_min > height * top_bottom_height[1] and \
                width - column_max < 5 and height - row_max < 5:
            self.uicompo_ = True
            return True
        return False

    def block_erase_from_bin(self, binary, pad):
        """
        Erases a rectangular block from a binary image with optional padding.
        
        This method removes a rectangular region from a binary image by drawing a filled
        black rectangle at the location specified by the bounding box of the current object.
        The rectangle can be expanded by a specified padding amount in all directions while
        respecting the image boundaries. This is useful for clearing detected UI elements or
        regions of interest from binary representations during element detection and analysis workflows.
        
        Args:
            binary: A binary image (numpy array or OpenCV Mat) from which the rectangular 
                block will be erased. The image is modified in-place.
            pad: The padding amount in pixels to expand the bounding box in all directions
                (top, bottom, left, right) before erasing. Prevents boundary violations by
                clamping coordinates to valid image dimensions.
        
        Returns:
            None. The method modifies the binary image in-place by setting the pixels in
            the specified rectangular region to 0 (black).
        """
        (column_min, row_min, column_max, row_max) = self.put_bbox()
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, binary.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, binary.shape[0])
        cv2.rectangle(binary, (column_min, row_min), (column_max, row_max), (0), -1)

