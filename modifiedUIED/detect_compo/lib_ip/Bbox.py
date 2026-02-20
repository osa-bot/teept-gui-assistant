import numpy as np
import detect_compo.lib_ip.ip_draw as draw


class Bbox:
    """
    Represents a bounding box with coordinate boundaries and spatial calculations.
    
        This class encapsulates a rectangular bounding box defined by minimum and maximum
        column and row coordinates. It provides functionality to calculate box dimensions,
        perform spatial operations such as intersection detection, merging, padding, and
        coordinate transformations.
    
        Attributes:
            col_min: The minimum column coordinate of the bounding box.
            row_min: The minimum row coordinate of the bounding box.
            col_max: The maximum column coordinate of the bounding box.
            row_max: The maximum row coordinate of the bounding box.
            width: The width of the bounding box, calculated as col_max - col_min.
            height: The height of the bounding box, calculated as row_max - row_min.
            box_area: The total area of the bounding box, calculated as width * height.
    
        Methods:
            - __init__: Initializes a bounding box object with coordinate boundaries and calculates dimensions.
            - put_bbox: Retrieves the bounding box coordinates as a tuple.
            - bbox_cal_area: Calculates and stores the area of the bounding box.
            - bbox_relation: Determines the spatial relationship between two bounding boxes.
            - bbox_relation_nms: Calculates the relation between two rectangles using NMS algorithm.
            - bbox_cvt_relative_position: Converts bounding box to relative position based on a base coordinator.
            - bbox_merge: Merges two intersected bounding boxes.
            - bbox_padding: Applies padding to the bounding box coordinates while respecting image boundaries.
    """

    def __init__(self, col_min, row_min, col_max, row_max):
        """
        Initializes a bounding box that defines a rectangular region on the screen for UI element tracking and interaction.
        
        This constructor establishes a bounding box using minimum and maximum column and row coordinates,
        which are essential for mapping screen regions to actionable areas in the task automation workflow.
        The dimensions are automatically computed to enable efficient spatial analysis and grid-based visual feedback.
        
        Args:
            col_min: The minimum column coordinate (left boundary) in pixels.
            row_min: The minimum row coordinate (top boundary) in pixels.
            col_max: The maximum column coordinate (right boundary) in pixels.
            row_max: The maximum row coordinate (bottom boundary) in pixels.
        
        Returns:
            None
        
        Attributes:
            col_min: The minimum column coordinate of the bounding box.
            row_min: The minimum row coordinate of the bounding box.
            col_max: The maximum column coordinate of the bounding box.
            row_max: The maximum row coordinate of the bounding box.
            width: The width of the bounding box in pixels, calculated as col_max - col_min.
            height: The height of the bounding box in pixels, calculated as row_max - row_min.
            box_area: The total area of the bounding box in square pixels, calculated as width * height.
        """
        self.col_min = col_min
        self.row_min = row_min
        self.col_max = col_max
        self.row_max = row_max

        self.width = col_max - col_min
        self.height = row_max - row_min
        self.box_area = self.width * self.height

    def put_bbox(self):
        """
        Retrieves the bounding box coordinates that define the spatial boundaries of a detected UI element.
        
        Returns the minimum and maximum column and row indices that establish the rectangular region
        occupied by this object on the screen. These coordinates are essential for mapping UI elements
        to the grid-based overlay system and enabling precise interaction guidance.
        
        Returns:
            tuple: A tuple containing four integers (col_min, row_min, col_max, row_max)
                representing the bounding box coordinates where col_min is the minimum
                column index, row_min is the minimum row index, col_max is the maximum
                column index, and row_max is the maximum row index.
        """
        return self.col_min, self.row_min, self.col_max, self.row_max

    def bbox_cal_area(self):
        """
        Calculates and stores the area of the bounding box.
        
        This method computes the area by multiplying the width and height of the
        bounding box. The calculated area is essential for determining the spatial
        extent of detected UI elements, which helps in prioritizing and filtering
        relevant interface components during task automation and element matching.
        
        Args:
            self: The instance of the Bbox class.
        
        Returns:
            float: The calculated area of the bounding box (width × height).
        """
        self.box_area = self.width * self.height
        return self.box_area

    def bbox_relation(self, bbox_b):
        """
        Determines the spatial relationship between two bounding boxes to support UI element detection and interaction guidance.
        
        This method analyzes how two rectangular regions relate to each other in screen space, which is essential for 
        identifying overlapping UI elements, determining containment relationships, and validating element positioning 
        during task automation workflows.
        
        Args:
            bbox_b (Bbox): The bounding box to compare against this bounding box (self).
        
        Returns:
            int: The spatial relationship between the two bounding boxes:
                -1 : This bounding box is completely contained within bbox_b
                0  : The bounding boxes do not intersect
                1  : bbox_b is completely contained within this bounding box
                2  : The bounding boxes are identical or partially intersect
        """
        col_min_a, row_min_a, col_max_a, row_max_a = self.put_bbox()
        col_min_b, row_min_b, col_max_b, row_max_b = bbox_b.put_bbox()

        # if a is in b
        if col_min_a > col_min_b and row_min_a > row_min_b and col_max_a < col_max_b and row_max_a < row_max_b:
            return -1
        # if b is in a
        elif col_min_a < col_min_b and row_min_a < row_min_b and col_max_a > col_max_b and row_max_a > row_max_b:
            return 1
        # a and b are non-intersect
        elif (col_min_a > col_max_b or row_min_a > row_max_b) or (col_min_b > col_max_a or row_min_b > row_max_a):
            return 0
        # intersection
        else:
            return 2

    def bbox_relation_nms(self, bbox_b, bias=(0, 0)):
        """
        Determine the spatial relationship between two bounding boxes to support UI element detection and overlap analysis.
        
        This method calculates intersection metrics (IoU, IoA, IoB) between two rectangles to identify
        whether they are disjoint, overlapping, or contained within each other. These relationships are
        essential for filtering redundant UI element detections and understanding element hierarchy in
        the interface layout.
        
        Args:
            bbox_b (Bbox): The second bounding box to compare against this bounding box.
            bias (tuple, optional): Bias values (col_bias, row_bias) to expand the bounding boxes
                before intersection calculation. Defaults to (0, 0).
        
        Returns:
            int: The spatial relationship between the two bounding boxes:
                - -1: This bounding box is contained within bbox_b
                - 0: The bounding boxes do not intersect
                - 1: bbox_b is contained within this bounding box
                - 2: The bounding boxes partially overlap
        """
        col_min_a, row_min_a, col_max_a, row_max_a = self.put_bbox()
        col_min_b, row_min_b, col_max_b, row_max_b = bbox_b.put_bbox()

        bias_col, bias_row = bias
        # get the intersected area
        col_min_s = max(col_min_a - bias_col, col_min_b - bias_col)
        row_min_s = max(row_min_a - bias_row, row_min_b - bias_row)
        col_max_s = min(col_max_a + bias_col, col_max_b + bias_col)
        row_max_s = min(row_max_a + bias_row, row_max_b + bias_row)
        w = np.maximum(0, col_max_s - col_min_s)
        h = np.maximum(0, row_max_s - row_min_s)
        inter = w * h
        area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
        area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
        iou = inter / (area_a + area_b - inter)
        ioa = inter / self.box_area
        iob = inter / bbox_b.box_area

        if iou == 0 and ioa == 0 and iob == 0:
            return 0

        # import lib_ip.ip_preprocessing as pre
        # org_iou, _ = pre.read_img('uied/data/input/7.jpg', 800)
        # print(iou, ioa, iob)
        # board = draw.draw_bounding_box(org_iou, [self], color=(255,0,0))
        # draw.draw_bounding_box(board, [bbox_b], color=(0,255,0), show=True)

        # contained by b
        if ioa >= 1:
            return -1
        # contains b
        if iob >= 1:
            return 1
        # not intersected with each other
        # intersected
        if iou >= 0.02 or iob > 0.2 or ioa > 0.2:
            return 2
        # if iou == 0:
        # print('ioa:%.5f; iob:%.5f; iou:%.5f' % (ioa, iob, iou))
        return 0

    def bbox_cvt_relative_position(self, col_min_base, row_min_base):
        '''
        Convert bounding box coordinates to relative positions within a specified region.
        
        This method adjusts the bounding box coordinates by adding base offsets, enabling
        the representation of UI element positions relative to a parent container or region
        of interest. This is essential for mapping detected elements to their actual screen
        locations when working with cropped or subdivided UI regions.
        
        Args:
            col_min_base (int): The column offset to add to the minimum column coordinate.
            row_min_base (int): The row offset to add to the minimum row coordinate.
        
        Returns:
            None: Modifies the bounding box coordinates in-place by updating col_min, col_max,
                  row_min, and row_max attributes.
        '''
        self.col_min += col_min_base
        self.col_max += col_min_base
        self.row_min += row_min_base
        self.row_max += row_min_base

    def bbox_merge(self, bbox_b):
        """
        Combine two overlapping bounding boxes into a single bounding box that encompasses both regions.
        
        This method calculates the union of two bounding boxes by finding the minimum and maximum
        coordinates across both boxes, creating a new bounding box that covers the entire area of
        both input boxes. This is useful for consolidating detected UI elements that overlap or
        are adjacent to each other.
        
        Args:
            bbox_b (Bbox): The second bounding box to merge with the current bounding box.
        
        Returns:
            Bbox: A new bounding box object representing the union of both input bounding boxes,
                  with coordinates spanning from the minimum top-left to the maximum bottom-right
                  of both boxes.
        """
        col_min_a, row_min_a, col_max_a, row_max_a = self.put_bbox()
        col_min_b, row_min_b, col_max_b, row_max_b = bbox_b.put_bbox()
        col_min = min(col_min_a, col_min_b)
        col_max = max(col_max_a, col_max_b)
        row_min = min(row_min_a, row_min_b)
        row_max = max(row_max_a, row_max_b)
        new_bbox = Bbox(col_min, row_min, col_max, row_max)
        return new_bbox

    def bbox_padding(self, image_shape, pad):
        """
        Expands the bounding box region with padding while constraining it within image boundaries.
        
        This method increases the bounding box dimensions by a specified padding amount in all directions,
        which is essential for creating expanded regions around detected UI elements to ensure sufficient
        context capture during element analysis and interaction. The padding is automatically clipped to
        prevent coordinates from exceeding the image dimensions, maintaining valid spatial references for
        screen region mapping.
        
        Args:
            image_shape: A tuple or array-like containing image dimensions. The first two elements
                represent the number of rows (height) and columns (width) used to establish boundary
                constraints for the padded bounding box.
            pad: An integer specifying the number of pixels to add to all sides of the bounding box.
        
        Returns:
            None. Modifies the bounding box coordinates in-place by updating col_min, col_max,
            row_min, and row_max attributes to their padded and boundary-constrained values.
        """
        row, col = image_shape[:2]
        self.col_min = max(self.col_min - pad, 0)
        self.col_max = min(self.col_max + pad, col)
        self.row_min = max(self.row_min - pad, 0)
        self.row_max = min(self.row_max + pad, row)