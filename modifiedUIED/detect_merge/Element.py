import numpy as np
import cv2


class Element:
    """
    Represents a spatial element with bounding box coordinates and categorical information.
    
        The Element class encapsulates an object with spatial positioning, category classification,
        and optional text content. It manages bounding box calculations, hierarchical relationships
        through parent-child connections, and provides utilities for spatial analysis, merging,
        intersection calculations, and visualization.
    
        Attributes:
            id: Unique identifier for the element.
            category: Category or type classification of the element.
            col_min: Minimum column coordinate of the element's bounding box.
            row_min: Minimum row coordinate of the element's bounding box.
            col_max: Maximum column coordinate of the element's bounding box.
            row_max: Maximum row coordinate of the element's bounding box.
            width: The width of the element calculated as col_max - col_min.
            height: The height of the element calculated as row_max - row_min.
            area: The total area of the element calculated as width * height.
            text_content: Optional text content associated with the element.
            parent_id: The identifier of the parent element.
            children: A list to store child elements.
    
        Methods:
            - __init__: Initializes an element with spatial and categorical information.
            - init_bound: Initializes the bounding box dimensions and area.
            - put_bbox: Retrieves the bounding box coordinates of the object.
            - wrap_info: Wraps the object's information into a dictionary representation.
            - resize: Resizes the bounding box dimensions by applying a resize ratio.
            - element_merge: Merges the current element with another element by combining their bounding boxes and text content.
            - calc_intersection_area: Calculates the intersection area and overlap metrics between this element and another element.
            - element_relation: Determines the spatial relationship between this element and another element.
            - visualize_element: Visualizes the element by drawing a bounding box on the provided image.
    """

    def __init__(self, id, corner, category, text_content=None):
        """
        Initializes an element representing a detected UI component with spatial and categorical information.
        
        This constructor creates an element object that encapsulates a UI component's positioning data, 
        category classification, and optional text content. By calculating spatial properties from bounding 
        box coordinates and establishing a hierarchical parent-child structure, it enables the system to 
        organize detected UI elements into a logical tree that reflects their relationships on the screen. 
        This structured representation is essential for matching user-provided descriptions to specific 
        UI components and providing accurate visual guidance through grid-based overlays.
        
        Args:
            id: A unique identifier for the element.
            corner: A tuple or sequence of four values representing the bounding box coordinates
                in the format (col_min, row_min, col_max, row_max).
            category: The category or type classification for this element (e.g., button, text field, image).
            text_content: Optional text content associated with the element. Defaults to None.
        
        Returns:
            None
        
        Attributes:
            id: Unique identifier for the element.
            category: Category or type classification of the element.
            col_min: Minimum column coordinate of the element's bounding box.
            row_min: Minimum row coordinate of the element's bounding box.
            col_max: Maximum column coordinate of the element's bounding box.
            row_max: Maximum row coordinate of the element's bounding box.
            width: The width of the element calculated as col_max - col_min.
            height: The height of the element calculated as row_max - row_min.
            area: The total area of the element calculated as width * height.
            text_content: Optional text content associated with the element.
            parent_id: The identifier of the parent element. Initially set to None.
            children: A list to store child elements. Initially an empty list.
        """
        self.id = id
        self.category = category
        self.col_min, self.row_min, self.col_max, self.row_max = corner
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

        self.text_content = text_content
        self.parent_id = None
        self.children = []  # list of elements

    def init_bound(self):
        """
        Establish spatial boundaries for UI element detection and interaction.
        
        This method computes the dimensional properties of a detected UI element's bounding box,
        which are essential for mapping screen regions to actionable areas and providing accurate
        visual guidance through the grid-based overlay interface.
        
        Args:
            self: The instance of the Element class.
        
        Returns:
            None
        
        Attributes:
            width: The horizontal span of the element, calculated as the difference between
                the maximum and minimum column coordinates.
            height: The vertical span of the element, calculated as the difference between
                the maximum and minimum row coordinates.
            area: The total spatial footprint of the element, calculated as the product of
                width and height, used for determining element prominence and interaction priority.
        """
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

    def put_bbox(self):
        """
        Retrieves the bounding box coordinates of the detected UI element.
        
        Returns the minimum and maximum column and row indices that define the 
        rectangular bounding box of this UI element on the screen. These coordinates 
        are essential for visual guidance and interaction mapping, allowing the system 
        to precisely locate and highlight UI components within the grid-based overlay 
        interface.
        
        Returns:
            tuple: A tuple containing four integers (col_min, row_min, col_max, row_max)
                representing the bounding box coordinates where col_min is the minimum
                column index, row_min is the minimum row index, col_max is the maximum
                column index, and row_max is the maximum row index.
        """
        return self.col_min, self.row_min, self.col_max, self.row_max

    def wrap_info(self):
        """
        Serializes the element's metadata into a dictionary for communication and analysis.
        
        This method converts the element's properties into a structured dictionary format that can be
        transmitted to the server or used for UI element matching and analysis. By organizing the element's
        identifier, classification, dimensions, and spatial coordinates into a standardized format, it enables
        the system to reference and locate UI components during task execution. The method conditionally includes
        text content and hierarchical relationships (parent/child references) when available, allowing the server
        to make informed decisions about element interactions and to match user-provided descriptions against
        detected UI components.
        
        Args:
            self: The instance of the Element class.
        
        Returns:
            dict: A dictionary containing the serialized element information with the following structure:
                - 'id': The unique identifier of the element.
                - 'class': The category classification of the element.
                - 'height': The height dimension of the element in pixels.
                - 'width': The width dimension of the element in pixels.
                - 'position': A nested dictionary containing spatial coordinates with keys
                  'column_min', 'row_min', 'column_max', and 'row_max'.
                - 'text_content': (Optional) The text content if it exists and is not None.
                - 'children': (Optional) A list of child element identifiers if children exist.
                - 'parent': (Optional) The parent element identifier if a parent exists.
        """
        info = {'id':self.id, 'class': self.category, 'height': self.height, 'width': self.width,
                'position': {'column_min': self.col_min, 'row_min': self.row_min, 'column_max': self.col_max,
                             'row_max': self.row_max}}
        if self.text_content is not None:
            info['text_content'] = self.text_content
        if len(self.children) > 0:
            info['children'] = []
            for child in self.children:
                info['children'].append(child.id)
        if self.parent_id is not None:
            info['parent'] = self.parent_id
        return info

    def resize(self, resize_ratio):
        """
        Resizes the bounding box dimensions by applying a resize ratio.
        
        This method scales all boundary coordinates (col_min, row_min, col_max, row_max)
        by the given resize ratio to adapt UI element positions to different screen resolutions
        or scaling factors. After scaling, it reinitializes the bounding box constraints to ensure
        the element's spatial properties remain consistent with the resized display dimensions.
        
        Args:
            resize_ratio (float): The scaling factor to apply to all boundary coordinates.
        
        Returns:
            None
        """
        self.col_min = int(self.col_min * resize_ratio)
        self.row_min = int(self.row_min * resize_ratio)
        self.col_max = int(self.col_max * resize_ratio)
        self.row_max = int(self.row_max * resize_ratio)
        self.init_bound()

    def element_merge(self, element_b, new_element=False, new_category=None, new_id=None):
        """
        Merges the current element with another element by combining their bounding boxes and text content.
        
        This method calculates the union of two elements' bounding boxes to create a larger region that encompasses both UI components. This is useful for grouping related interface elements together, enabling more efficient interaction guidance and element tracking during task automation workflows.
        
        The merged bounding box represents the minimum and maximum extents in both dimensions, and text content from both elements is combined to preserve all relevant information about the merged region.
        
        Args:
            element_b (Element): The element to merge with the current element.
            new_element (bool, optional): If True, returns a new Element instance with merged properties
                instead of modifying the current element in place. Defaults to False.
            new_category (str, optional): The category to assign to the new element if new_element is True.
            new_id (str, optional): The identifier to assign to the new element if new_element is True.
        
        Returns:
            Element or None: If new_element is True, returns a new Element instance with the merged
                bounding box and specified category and id. If new_element is False,
                modifies the current element's bounding box in place and returns None.
        """
        col_min_a, row_min_a, col_max_a, row_max_a = self.put_bbox()
        col_min_b, row_min_b, col_max_b, row_max_b = element_b.put_bbox()
        new_corner = (min(col_min_a, col_min_b), min(row_min_a, row_min_b), max(col_max_a, col_max_b), max(row_max_a, row_max_b))
        if element_b.text_content is not None:
            self.text_content = element_b.text_content if self.text_content is None else self.text_content + '\n' + element_b.text_content
        if new_element:
            return Element(new_id, new_corner, new_category)
        else:
            self.col_min, self.row_min, self.col_max, self.row_max = new_corner
            self.init_bound()

    def calc_intersection_area(self, element_b, bias=(0, 0)):
        """
        Calculates the intersection area and overlap metrics between this element and another element.
        
        This method computes the bounding box intersection between two elements to quantify their spatial
        overlap, which is essential for matching detected UI components and determining if elements occupy
        the same screen region. It derives multiple intersection-over-union (IoU) metrics that measure how
        well two elements align, enabling accurate element identification and interaction verification.
        
        Args:
            element_b: Another element object with which to calculate intersection metrics.
            bias: A tuple of two values (col_bias, row_bias) representing column and row bias adjustments
                applied to the minimum coordinates of the intersection calculation. Defaults to (0, 0).
        
        Returns:
            A tuple containing four float values:
                - inter: The intersection area between the two bounding boxes.
                - iou: Intersection over union, calculated as the intersection area divided
                    by the union of both elements' areas.
                - ioa: Intersection over area of this element, calculated as the intersection
                    area divided by this element's area.
                - iob: Intersection over area of the other element, calculated as the
                    intersection area divided by the other element's area.
        """
        a = self.put_bbox()
        b = element_b.put_bbox()
        col_min_s = max(a[0], b[0]) - bias[0]
        row_min_s = max(a[1], b[1]) - bias[1]
        col_max_s = min(a[2], b[2])
        row_max_s = min(a[3], b[3])
        w = np.maximum(0, col_max_s - col_min_s)
        h = np.maximum(0, row_max_s - row_min_s)
        inter = w * h

        iou = inter / (self.area + element_b.area - inter)
        ioa = inter / self.area
        iob = inter / element_b.area

        return inter, iou, ioa, iob

    def element_relation(self, element_b, bias=(0, 0)):
        """
        Determines the spatial relationship between two UI elements to understand their positioning and overlap.
        
        This method analyzes how two elements interact spatially, which is essential for understanding the layout
        of the user interface and determining if elements are nested, overlapping, or completely separate.
        This information helps in correctly identifying and interacting with UI components during task execution.
        
        Args:
            element_b (Element): The second element to compare with the current element.
            bias (tuple): A tuple of (horizontal_bias, vertical_bias) to adjust the comparison coordinates.
                          Defaults to (0, 0).
        
        Returns:
            int: The spatial relationship between the elements:
                 -1 : Current element is contained within element_b
                 0  : Elements do not intersect
                 1  : element_b is contained within the current element
                 2  : Elements are identical or partially intersected
        """
        inter, iou, ioa, iob = self.calc_intersection_area(element_b, bias)

        # area of intersection is 0
        if ioa == 0:
            return 0
        # a in b
        if ioa >= 1:
            return -1
        # b in a
        if iob >= 1:
            return 1
        return 2

    def visualize_element(self, img, color=(0, 255, 0), line=1, show=False):
        """
        Visualizes the detected UI element by drawing a bounding box on the provided image.
        
        This method renders a rectangular bounding box around the element's location on the given
        image, enabling visual verification of element detection accuracy. Optionally, it can display 
        the annotated image in a window for immediate inspection during debugging or task execution.
        
        Args:
            img: The image on which to draw the bounding box.
            color: The color of the bounding box in BGR format as a tuple of three
                integers representing blue, green, and red channels respectively.
                Defaults to (0, 255, 0) for green.
            line: The thickness of the bounding box line in pixels. Defaults to 1.
            show: A boolean flag indicating whether to display the image in a window
                after drawing the bounding box. Defaults to False.
        
        Returns:
            None. The method modifies the input image in-place by drawing the
            bounding box on it.
        """
        loc = self.put_bbox()
        cv2.rectangle(img, loc[:2], loc[2:], color, line)
        # for child in self.children:
        #     child.visualize_element(img, color=(255, 0, 255), line=line)
        if show:
            cv2.imshow('element', img)
            cv2.waitKey(0)
            cv2.destroyWindow('element')
