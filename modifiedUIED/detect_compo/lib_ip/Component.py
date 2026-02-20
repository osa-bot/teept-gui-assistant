from detect_compo.lib_ip.Bbox import Bbox
import detect_compo.lib_ip.ip_draw as draw

import cv2


def cvt_compos_relative_pos(compos, col_min_base, row_min_base):
    """
    Converts the absolute positions of UI components to relative positions within a normalized coordinate space.
    
    This method iterates through a collection of detected UI components and updates each
    component's position to be relative to a specified base point. This normalization is essential
    for creating a consistent coordinate system that can be mapped to visual grid overlays and
    used for interactive guidance, regardless of the original screen dimensions or detection offsets.
    
    Args:
        compos: A collection of component objects whose positions need to be converted.
        col_min_base: The minimum column value to use as the base reference point for
            horizontal position conversion.
        row_min_base: The minimum row value to use as the base reference point for
            vertical position conversion.
    
    Returns:
        None. The method modifies the components in-place by updating their positions
        to be relative to the specified base coordinates.
    """
    for compo in compos:
        compo.compo_relative_position(col_min_base, row_min_base)


def compos_containment(compos):
    """
    Establishes containment hierarchy among composite UI elements based on their spatial relationships.
    
    This method determines which composite objects are contained within others by analyzing
    pairwise spatial or hierarchical relations. For each pair of composites, it evaluates
    their mutual relationship and records containment by appending the index of the contained
    object to the container's containment list. This enables the system to understand the
    nested structure of UI components, which is essential for accurate element detection and
    interaction guidance.
    
    Args:
        compos: A list of composite objects representing UI elements, each having a 
            compo_relation method that returns the containment relationship (-1, 0, or 1)
            and a contain attribute (list) to store indices of objects contained within it.
    
    Returns:
        None. The method modifies the contain lists of the composite objects in-place,
        establishing the hierarchical containment relationships between UI elements.
    """
    for i in range(len(compos) - 1):
        for j in range(i + 1, len(compos)):
            relation = compos[i].compo_relation(compos[j])
            if relation == -1:
                compos[j].contain.append(i)
            if relation == 1:
                compos[i].contain.append(j)


def compos_update(compos, org_shape):
    """
    Initializes detected UI components with sequential identifiers and dimensional metadata for task automation.
    
    This method prepares identified UI elements for interaction by assigning each component a unique index
    and storing the original screen dimensions. Sequential IDs enable the system to reference and track
    specific UI elements during task execution, while the shape information maintains the coordinate space
    needed for accurate element location and interaction guidance.
    
    Args:
        compos: A collection of detected UI component objects to be initialized.
        org_shape: A tuple or array representing the original screen dimensions (height, width) 
                   to be associated with each component for coordinate mapping.
    
    Returns:
        None. Modifies each component in-place by calling its update method with the assigned 
        index and shape information.
    """
    for i, compo in enumerate(compos):
        # start from 1, id 0 is background
        compo.compo_update(i + 1, org_shape)


class Component:
    """
    Represents a visual component detected in an image with geometric and structural properties.
    
        The Component class encapsulates information about a detected region in an image, including
        its boundaries, bounding box, dimensions, and relationships with other components. It provides
        methods for updating component properties, analyzing geometric characteristics, performing
        spatial operations, and managing component hierarchies.
    
        Attributes:
            id: Unique identifier for the component.
            region: The region data representing the component's pixels or coordinates.
            boundary: The boundary coordinates of the component extracted from the region.
            bbox: The bounding box object containing the component's rectangular bounds.
            bbox_area: The area of the bounding box.
            region_area: The total number of pixels in the region.
            width: The width of the component's boundary.
            height: The height of the component's boundary.
            image_shape: The shape/dimensions of the parent image.
            area: The total area calculated as width multiplied by height.
            category: The classification category of the component.
            contain: A list to store child components or contained elements.
            rect_: Rectangle representation of the component.
            line_: Line representation of the component.
            redundant: Boolean flag indicating if the component is redundant.
    
        Methods:
            - __init__: Initializes a component object with region and image information.
            - compo_update: Updates the component with new identification and shape information.
            - put_bbox: Puts the bounding box.
            - compo_update_bbox_area: Updates the bounding box area of the component.
            - compo_get_boundary: Gets the bounding boundary of the component.
            - compo_get_bbox: Gets the top left and bottom right points of boundary.
            - compo_is_rectangle: Detects if the component is a rectangle by analyzing border evenness and dents.
            - compo_is_line: Checks if the component is a line by examining its boundary.
            - compo_relation: Determines the spatial relationship between this component and another.
            - compo_relative_position: Converts to relative position based on a base coordinator.
            - compo_merge: Merges another component into the current component.
            - compo_clipping: Clips an image to the bounding box region of the component with optional padding.
    """

    def __init__(self, region, image_shape):
        """
        Initializes a Component object by extracting and analyzing UI element properties from a detected region.
        
        This constructor creates a component instance that represents a detected UI element by analyzing 
        the provided region data and computing its geometric properties. These properties are essential 
        for identifying, locating, and interacting with UI elements during task automation workflows.
        
        Args:
            region: The region data representing the component's pixels or coordinates detected in the image.
            image_shape: The shape/dimensions (height, width) of the parent image containing this component.
        
        Returns:
            None
        
        Attributes:
            id: Unique identifier for the component (initialized as None, assigned during processing).
            region: The region data representing the component's pixels or coordinates.
            boundary: The boundary coordinates of the component extracted from the region.
            bbox: The bounding box object containing the component's rectangular bounds.
            bbox_area: The area of the bounding box in pixels.
            region_area: The total number of pixels in the detected region.
            width: The width of the component's boundary.
            height: The height of the component's boundary.
            image_shape: The shape/dimensions of the parent image.
            area: The total area calculated as width multiplied by height.
            category: The classification category of the component (set to 'Compo').
            contain: A list to store child components or contained elements within this component.
            rect_: Rectangle representation of the component for visualization (initialized as None).
            line_: Line representation of the component for visualization (initialized as None).
            redundant: Boolean flag indicating if the component is redundant or duplicate (initialized as False).
        """
        self.id = None
        self.region = region
        self.boundary = self.compo_get_boundary()
        self.bbox = self.compo_get_bbox()
        self.bbox_area = self.bbox.box_area

        self.region_area = len(region)
        self.width = len(self.boundary[0])
        self.height = len(self.boundary[2])
        self.image_shape = image_shape
        self.area = self.width * self.height

        self.category = 'Compo'
        self.contain = []

        self.rect_ = None
        self.line_ = None
        self.redundant = False

    def compo_update(self, id, org_shape):
        """
        Reinitializes the component's properties to reflect its current state in the UI.
        
        This method updates the component's identification and synchronizes its dimensional
        attributes with the actual bounding box measurements. By recalculating width, height,
        and area properties, it ensures the component's metadata accurately represents its
        position and size on the screen, which is essential for precise UI element detection
        and interaction guidance.
        
        Args:
            id: The unique identifier for the component.
            org_shape: The original shape dimensions of the component (typically the screenshot dimensions).
        
        Returns:
            None
        """
        self.id = id
        self.image_shape = org_shape
        self.width = self.bbox.width
        self.height = self.bbox.height
        self.bbox_area = self.bbox.box_area
        self.area = self.width * self.height

    def put_bbox(self):
        """
        Persists the bounding box visualization to the display.
        
        This method delegates the rendering operation to the internal bounding box object,
        ensuring that the component's spatial boundaries are visually represented on the screen
        for user guidance and interaction mapping.
        
        Returns:
            The result of persisting the bounding box from the internal bbox object.
        """
        return self.bbox.put_bbox()

    def compo_update_bbox_area(self):
        """
        Calculates and caches the spatial dimensions of the component's bounding box.
        
        This method computes the area occupied by the component's bounding box region,
        which is essential for spatial analysis and component prioritization during UI
        element detection and matching operations. The computed area is stored for
        efficient access during subsequent component comparisons and filtering.
        
        Args:
            None
        
        Returns:
            None
        """
        self.bbox_area = self.bbox.bbox_cal_area()

    def compo_get_boundary(self):
        """
        Compute the spatial boundaries of a UI component by analyzing its pixel region.
        
        Detects the bounding edges of a component across all four directions by tracking
        the minimum and maximum coordinates for each row and column. This enables precise
        localization of UI elements within the screen for interaction mapping and visual guidance.
        
        The boundary structure organizes edge information by coordinate axis:
        - Top/Bottom boundaries: map each column index to its minimum/maximum row position
        - Left/Right boundaries: map each row index to its minimum/maximum column position
        
        Args:
            None (operates on self.region - the set of pixel coordinates comprising the component)
        
        Returns:
            list: A list of four sorted boundary dictionaries [top, bottom, left, right], where:
                - top (dict): Column indices mapped to minimum row values
                - bottom (dict): Column indices mapped to maximum row values
                - left (dict): Row indices mapped to minimum column values
                - right (dict): Row indices mapped to maximum column values
                Each dictionary is sorted by coordinate index for consistent ordering.
        """
        border_up, border_bottom, border_left, border_right = {}, {}, {}, {}
        for point in self.region:
            # point: (row_index, column_index)
            # up, bottom: (column_index, min/max row border) detect range of each column
            if point[1] not in border_up or border_up[point[1]] > point[0]:
                border_up[point[1]] = point[0]
            if point[1] not in border_bottom or border_bottom[point[1]] < point[0]:
                border_bottom[point[1]] = point[0]
            # left, right: (row_index, min/max column border) detect range of each row
            if point[0] not in border_left or border_left[point[0]] > point[1]:
                border_left[point[0]] = point[1]
            if point[0] not in border_right or border_right[point[0]] < point[1]:
                border_right[point[0]] = point[1]

        boundary = [border_up, border_bottom, border_left, border_right]
        # descending sort
        for i in range(len(boundary)):
            boundary[i] = [[k, boundary[i][k]] for k in boundary[i].keys()]
            boundary[i] = sorted(boundary[i], key=lambda x: x[0])
        return boundary

    def compo_get_bbox(self):
        """
        Calculate the bounding box coordinates that encompass all detected UI component boundaries.
        
        This method extracts the minimal rectangular region containing the component by analyzing
        its boundary data across horizontal and vertical axes. The resulting bounding box enables
        precise spatial mapping of UI elements for interaction guidance and visual feedback.
        
        Args:
            self: Component instance with boundary data containing top, bottom, left, and right
                  edge information in the format [top, bottom, left, right] where:
                  - top, bottom: tuples of (column_index, row_border_value)
                  - left, right: tuples of (row_index, column_border_value)
        
        Returns:
            Bbox: A bounding box object with coordinates (col_min, row_min, col_max, row_max)
                  representing the top-left and bottom-right corners of the component's
                  rectangular region in screen space.
        """
        col_min, row_min = (int(min(self.boundary[0][0][0], self.boundary[1][-1][0])), int(min(self.boundary[2][0][0], self.boundary[3][-1][0])))
        col_max, row_max = (int(max(self.boundary[0][0][0], self.boundary[1][-1][0])), int(max(self.boundary[2][0][0], self.boundary[3][-1][0])))
        bbox = Bbox(col_min, row_min, col_max, row_max)
        return bbox

    def compo_is_rectangle(self, min_rec_evenness, max_dent_ratio, test=False):
        """
        Validate whether a component represents a rectangle by analyzing the regularity and deformations of its borders.
        
        This method examines each border of the component to determine if it maintains the geometric properties
        of a rectangle. It detects irregularities such as dents, pits, and abnormal surface changes that would
        indicate the component is not a valid rectangle. This validation is essential for accurately identifying
        and classifying UI elements during component detection and analysis.
        
        Args:
            min_rec_evenness (float): Minimum threshold for border evenness ratio (0-1). Lower values are more 
                permissive. Automatically adjusted to 0.85 for large components (height > 30% of image).
            max_dent_ratio (float): Maximum acceptable ratio of dented/pitted pixels relative to border length (0-1).
                Borders exceeding this ratio indicate the component is not rectangular.
            test (bool, optional): If True, enables debug output and visualization of boundary analysis. Defaults to False.
        
        Returns:
            bool: True if the component is classified as a rectangle, False otherwise. Also sets the instance
                attribute rect_ with the same boolean value.
        """
        dent_direction = [1, -1, 1, -1]  # direction for convex

        flat = 0
        parameter = 0
        for n, border in enumerate(self.boundary):
            parameter += len(border)
            # dent detection
            pit = 0  # length of pit
            depth = 0  # the degree of surface changing
            if n <= 1:
                adj_side = max(len(self.boundary[2]), len(self.boundary[3]))  # get maximum length of adjacent side
            else:
                adj_side = max(len(self.boundary[0]), len(self.boundary[1]))

            # -> up, bottom: (column_index, min/max row border)
            # -> left, right: (row_index, min/max column border) detect range of each row
            abnm = 0
            for i in range(int(3 + len(border) * 0.02), len(border) - 1):
                # calculate gradient
                difference = border[i][1] - border[i + 1][1]
                # the degree of surface changing
                depth += difference
                # ignore noise at the start of each direction
                if i / len(border) < 0.08 and (dent_direction[n] * difference) / adj_side > 0.5:
                    depth = 0  # reset

                # print(border[i][1], i / len(border), depth, (dent_direction[n] * difference) / adj_side)
                # if the change of the surface is too large, count it as part of abnormal change
                if abs(depth) / adj_side > 0.3:
                    abnm += 1  # count the size of the abnm
                    # if the abnm is too big, the shape should not be a rectangle
                    if abnm / len(border) > 0.1:
                        if test:
                            print('abnms', abnm, abnm / len(border))
                            draw.draw_boundary([self], self.image_shape, show=True)
                        self.rect_ = False
                        return False
                    continue
                else:
                    # reset the abnm if the depth back to normal
                    abnm = 0

                # if sunken and the surface changing is large, then counted as pit
                if dent_direction[n] * depth < 0 and abs(depth) / adj_side > 0.15:
                    pit += 1
                    continue

                # if the surface is not changing to a pit and the gradient is zero, then count it as flat
                if abs(depth) < 1 + adj_side * 0.015:
                    flat += 1
                if test:
                    print(depth, adj_side, flat)
            # if the pit is too big, the shape should not be a rectangle
            if pit / len(border) > max_dent_ratio:
                if test:
                    print('pit', pit, pit / len(border))
                    draw.draw_boundary([self], self.image_shape, show=True)
                self.rect_ = False
                return False
        if test:
            print(flat / parameter, '\n')
            draw.draw_boundary([self], self.image_shape, show=True)
        # ignore text and irregular shape
        if self.height / self.image_shape[0] > 0.3:
            min_rec_evenness = 0.85
        if (flat / parameter) < min_rec_evenness:
            self.rect_ = False
            return False
        self.rect_ = True
        return True

    def compo_is_line(self, min_line_thickness):
        """
        Determine if this component represents a line by analyzing its geometric boundaries.
        
        This method evaluates whether the component is predominantly linear (either horizontal or vertical)
        by examining the thickness consistency across its extent. A component is classified as a line if
        the majority of its cross-sectional measurements fall below the specified thickness threshold,
        which is essential for distinguishing UI separators and borders from other rectangular elements.
        
        Args:
            min_line_thickness (int): Maximum allowed thickness (in pixels) for the component to be
                                      considered a line. Components with consistent thin profiles are
                                      more likely to be lines.
        
        Returns:
            bool: True if the component is identified as a line (either horizontal or vertical),
                  False otherwise. Also sets the instance variable `line_` to reflect this classification.
        """
        # horizontally
        slim = 0
        for i in range(self.width):
            if abs(self.boundary[1][i][1] - self.boundary[0][i][1]) <= min_line_thickness:
                slim += 1
        if slim / len(self.boundary[0]) > 0.93:
            self.line_ = True
            return True
        # vertically
        slim = 0
        for i in range(self.height):
            if abs(self.boundary[2][i][1] - self.boundary[3][i][1]) <= min_line_thickness:
                slim += 1
        if slim / len(self.boundary[2]) > 0.93:
            self.line_ = True
            return True
        self.line_ = False
        return False

    def compo_relation(self, compo_b, bias=(0, 0)):
        """
        Determines the spatial relationship between this component and another component's bounding boxes.
        
        This method evaluates how two UI components are positioned relative to each other,
        which is essential for understanding the layout structure and spatial organization of
        detected GUI elements during task automation and element matching.
        
        Args:
            compo_b (Component): The other component to compare spatial relationship with.
            bias (tuple): Optional coordinate offset (x, y) to apply during comparison. Defaults to (0, 0).
        
        Returns:
            int: Spatial relationship code:
                -1 : compo_b completely contains this component
                0  : components do not intersect
                1  : this component completely contains compo_b
                2  : components are identical or partially intersect
        """
        return self.bbox.bbox_relation_nms(compo_b.bbox, bias)

    def compo_relative_position(self, col_min_base, row_min_base):
        '''
        Convert component's bounding box to relative coordinates within a parent container.
        
        This method adjusts the component's position relative to a base coordinate system,
        enabling proper spatial representation when components are nested or positioned
        within larger UI layouts. This is essential for maintaining accurate element
        positioning during screenshot analysis and grid-based visual guidance.
        
        Args:
            col_min_base: Minimum column coordinate of the base container
            row_min_base: Minimum row coordinate of the base container
        
        Returns:
            None (modifies the component's bounding box in-place)
        '''
        self.bbox.bbox_cvt_relative_position(col_min_base, row_min_base)

    def compo_merge(self, compo_b):
        """
        Merges another component into the current component to consolidate overlapping or adjacent UI elements.
        
        This method combines the bounding boxes of two components and updates the component's internal 
        state to reflect the unified region. This is essential for accurately representing merged UI 
        elements during the detection and analysis phase, ensuring that related components are properly 
        consolidated into a single entity for interaction guidance.
        
        Args:
            compo_b (Component): The component to merge with the current component.
        
        Returns:
            None
        """
        self.bbox = self.bbox.bbox_merge(compo_b.bbox)
        self.compo_update(self.id, self.image_shape)

    def compo_clipping(self, img, pad=0, show=False):
        """
        Extracts a visual region of a detected UI component from a screenshot for analysis and verification.
        
        This method isolates a rectangular region from the input image based on the component's 
        bounding box coordinates, with optional padding to capture surrounding context. This is 
        essential for verifying component detection accuracy, performing detailed visual analysis, 
        and providing users with focused visual feedback about identified UI elements during task 
        execution.
        
        Args:
            img: The input image (screenshot) from which to extract the component region.
            pad: The padding size in pixels to expand the bounding box in all directions.
                Defaults to 0 (no padding). Padding is automatically constrained by image boundaries
                to prevent out-of-bounds access.
            show: A boolean flag indicating whether to display the clipped region in a window
                for debugging or verification purposes. Defaults to False.
        
        Returns:
            A numpy array containing the clipped image region corresponding to the component's
            bounding box with applied padding, suitable for further analysis or display.
        """
        (column_min, row_min, column_max, row_max) = self.put_bbox()
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, img.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, img.shape[0])
        clip = img[row_min:row_max, column_min:column_max]
        if show:
            cv2.imshow('clipping', clip)
            cv2.waitKey()
        return clip
