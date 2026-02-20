import cv2
import numpy as np


class Text:
    '''
    *********************
        *** Visualization ***
        *********************
    '''
    def __init__(self, id, content, location):
        """
        Initialize a text element by capturing its content and spatial properties for UI analysis.
        
        This constructor creates a text element object that represents detected text in the UI by
        storing its identification, content, and bounding box coordinates. It automatically computes
        geometric properties (width, height, area) and character-level metrics that are essential
        for matching user descriptions to UI components and determining element positioning within
        the interactive grid interface.
        
        Args:
            id: The unique identifier for this text element.
            content: The text content or string value of this element.
            location: A dictionary containing the spatial coordinates with keys
                'left', 'right', 'top', and 'bottom' representing the bounding box
                of the text element on the screen.
        
        Attributes:
            id: The unique identifier for this text element.
            content: The text content or string value of this element.
            location: Dictionary containing the spatial coordinates of the element's
                bounding box.
            width: The horizontal span of the element calculated as the difference
                between right and left coordinates.
            height: The vertical span of the element calculated as the difference
                between bottom and top coordinates.
            area: The total area of the element's bounding box calculated as width
                multiplied by height.
            word_width: The average width per character calculated by dividing the
                total width by the number of characters in the content.
        """
        self.id = id
        self.content = content
        self.location = location

        self.width = self.location['right'] - self.location['left']
        self.height = self.location['bottom'] - self.location['top']
        self.area = self.width * self.height
        self.word_width = self.width / len(self.content)

    '''
    ********************************
    *** Relation with Other text ***
    ********************************
    '''
    def is_justified(self, ele_b, direction='h', max_bias_justify=4):
        """
        Determine if two text elements are spatially aligned based on their directional relationship.
        
        This method validates whether two UI elements maintain proper alignment along a specified axis,
        which is essential for accurately grouping and understanding the layout structure of detected
        text elements on the screen. Elements that are justified share common edge boundaries within
        a tolerance threshold, indicating they are intentionally positioned together as part of the
        same logical content block.
        
        Args:
            ele_b: The second text element to compare alignment with.
            direction (str): The axis along which to check alignment.
                - 'v': Vertical alignment - checks if left and right edges are justified
                - 'h': Horizontal alignment - checks if top and bottom edges are justified
                Defaults to 'h'.
            max_bias_justify (int): Maximum allowed pixel deviation between corresponding edges
                for elements to be considered justified. Defaults to 4.
        
        Returns:
            bool: True if the elements are justified along the specified direction within the
                tolerance threshold, False otherwise.
        """
        l_a = self.location
        l_b = ele_b.location
        # connected vertically - up and below
        if direction == 'v':
            # left and right should be justified
            if abs(l_a['left'] - l_b['left']) < max_bias_justify and abs(l_a['right'] - l_b['right']) < max_bias_justify:
                return True
            return False
        elif direction == 'h':
            # top and bottom should be justified
            if abs(l_a['top'] - l_b['top']) < max_bias_justify and abs(l_a['bottom'] - l_b['bottom']) < max_bias_justify:
                return True
            return False

    def is_on_same_line(self, text_b, direction='h', bias_gap=4, bias_justify=4):
        '''
        Determine if two text elements are aligned on the same row or column, enabling proper grouping and sequencing of UI components during layout analysis.
        
        This method checks spatial alignment between text elements to identify connected components that form logical groups within the UI structure. By validating both positional alignment and proximity, it ensures accurate detection of related UI elements that should be processed together.
        
        Args:
            text_b (Text): The text element to compare alignment with.
            direction (str, optional): The alignment direction to check.
                - 'h': horizontal alignment (same row, left-right connection)
                - 'v': vertical alignment (same column, up-down connection)
                Defaults to 'h'.
            bias_gap (int, optional): Maximum acceptable gap (in pixels) between element edges for considering them connected. Defaults to 4.
            bias_justify (int, optional): Maximum acceptable deviation (in pixels) for justification alignment check. Defaults to 4.
        
        Returns:
            bool: True if the elements are aligned on the same line in the specified direction with acceptable gap and justification, False otherwise.
        '''
        l_a = self.location
        l_b = text_b.location
        # connected vertically - up and below
        if direction == 'v':
            # left and right should be justified
            if self.is_justified(text_b, direction='v', max_bias_justify=bias_justify):
                # top and bottom should be connected (small gap)
                if abs(l_a['bottom'] - l_b['top']) < bias_gap or abs(l_a['top'] - l_b['bottom']) < bias_gap:
                    return True
            return False
        elif direction == 'h':
            # top and bottom should be justified
            if self.is_justified(text_b, direction='h', max_bias_justify=bias_justify):
                # top and bottom should be connected (small gap)
                if abs(l_a['right'] - l_b['left']) < bias_gap or abs(l_a['left'] - l_b['right']) < bias_gap:
                    return True
            return False

    def is_intersected(self, text_b, bias):
        """
        Determines whether this text object intersects with another text object.
        
        Checks if the bounding boxes of two text objects overlap by calculating
        the intersection area. This is useful for identifying UI elements that occupy
        overlapping screen regions during element detection and analysis. An optional
        bias value can be applied to adjust the intersection threshold, allowing for
        flexible matching of text elements that may be in close proximity.
        
        Args:
            text_b (Text): Another text object to check intersection against.
            bias (int): A bias value to adjust the left and top boundaries of the
                intersection calculation, effectively expanding or contracting the
                intersection threshold for more flexible element matching.
        
        Returns:
            bool: True if the text objects' bounding boxes intersect with an area
                greater than zero, False otherwise.
        """
        l_a = self.location
        l_b = text_b.location
        left_in = max(l_a['left'], l_b['left']) + bias
        top_in = max(l_a['top'], l_b['top']) + bias
        right_in = min(l_a['right'], l_b['right'])
        bottom_in = min(l_a['bottom'], l_b['bottom'])

        w_in = max(0, right_in - left_in)
        h_in = max(0, bottom_in - top_in)
        area_in = w_in * h_in
        if area_in > 0:
            return True

    '''
    ***********************
    *** Revise the Text ***
    ***********************
    '''
    def merge_text(self, text_b):
        """
        Merges another text object with the current text object to consolidate detected UI elements.
        
        This method combines two text objects by calculating the bounding box that encompasses 
        both objects, concatenating their content in left-to-right order, and updating all spatial 
        properties accordingly. This is useful when OCR or text detection identifies what should be 
        a single UI element as multiple separate text regions, allowing them to be unified for 
        accurate element matching and interaction.
        
        Args:
            text_b (Text): Another text object to merge with the current text object.
        
        Returns:
            None. The method modifies the current object's properties in place, updating its 
            location dictionary (left, top, right, bottom), width, height, area, content 
            (concatenated with space separator), and word_width based on the merged result 
            of both text objects.
        """
        text_a = self
        top = min(text_a.location['top'], text_b.location['top'])
        left = min(text_a.location['left'], text_b.location['left'])
        right = max(text_a.location['right'], text_b.location['right'])
        bottom = max(text_a.location['bottom'], text_b.location['bottom'])
        self.location = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        self.width = self.location['right'] - self.location['left']
        self.height = self.location['bottom'] - self.location['top']
        self.area = self.width * self.height

        left_element = text_a
        right_element = text_b
        if text_a.location['left'] > text_b.location['left']:
            left_element = text_b
            right_element = text_a
        self.content = left_element.content + ' ' + right_element.content
        self.word_width = self.width / len(self.content)

    def shrink_bound(self, binary_map):
        """
        Refines the bounding box boundaries to precisely encompass only the detected content by eliminating empty rows and columns from all edges.
        
        This method processes a binary representation of the screen region to identify and exclude empty (all-zero) rows and columns from the top, bottom, left, and right edges of the current bounding box. By tightening the spatial boundaries around actual content, it ensures accurate element localization for subsequent UI interaction and analysis tasks. The method updates the location boundaries and recalculates the associated dimensional properties.
        
        Args:
            binary_map: A 2D binary numpy array where non-zero values represent detected content and zero values represent empty space.
        
        Returns:
            None. The method modifies the object's location dictionary (top, bottom, left, right) in place and updates the width, height, area, and word_width attributes accordingly.
        """
        bin_clip = binary_map[self.location['top']:self.location['bottom'], self.location['left']:self.location['right']]
        height, width = np.shape(bin_clip)

        shrink_top = 0
        shrink_bottom = 0
        for i in range(height):
            # top
            if shrink_top == 0:
                if sum(bin_clip[i]) == 0:
                    shrink_top = 1
                else:
                    shrink_top = -1
            elif shrink_top == 1:
                if sum(bin_clip[i]) != 0:
                    self.location['top'] += i
                    shrink_top = -1
            # bottom
            if shrink_bottom == 0:
                if sum(bin_clip[height-i-1]) == 0:
                    shrink_bottom = 1
                else:
                    shrink_bottom = -1
            elif shrink_bottom == 1:
                if sum(bin_clip[height-i-1]) != 0:
                    self.location['bottom'] -= i
                    shrink_bottom = -1

            if shrink_top == -1 and shrink_bottom == -1:
                break

        shrink_left = 0
        shrink_right = 0
        for j in range(width):
            # left
            if shrink_left == 0:
                if sum(bin_clip[:, j]) == 0:
                    shrink_left = 1
                else:
                    shrink_left = -1
            elif shrink_left == 1:
                if sum(bin_clip[:, j]) != 0:
                    self.location['left'] += j
                    shrink_left = -1
            # right
            if shrink_right == 0:
                if sum(bin_clip[:, width-j-1]) == 0:
                    shrink_right = 1
                else:
                    shrink_right = -1
            elif shrink_right == 1:
                if sum(bin_clip[:, width-j-1]) != 0:
                    self.location['right'] -= j
                    shrink_right = -1

            if shrink_left == -1 and shrink_right == -1:
                break
        self.width = self.location['right'] - self.location['left']
        self.height = self.location['bottom'] - self.location['top']
        self.area = self.width * self.height
        self.word_width = self.width / len(self.content)

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def visualize_element(self, img, color=(0, 0, 255), line=1, show=False):
        """
        Visualizes a detected UI element by drawing a bounding rectangle on a screenshot.
        
        This method highlights the location of a text element on the provided image by drawing a rectangle
        around its detected boundaries. This visual feedback helps users identify which UI component has been
        located and analyzed by the system. Optionally, the method can display the element's extracted content
        and render the annotated image for inspection.
        
        Args:
            img: The screenshot image on which to draw the rectangle.
            color: The color of the rectangle in BGR format. Defaults to red (0, 0, 255).
            line: The thickness of the rectangle's border line in pixels. Defaults to 1.
            show: Whether to print the element's extracted content and display the annotated image in a window. Defaults to False.
        
        Returns:
            None. The method modifies the input image in-place by drawing the rectangle on it.
        """
        loc = self.location
        cv2.rectangle(img, (loc['left'], loc['top']), (loc['right'], loc['bottom']), color, line)
        if show:
            print(self.content)
            cv2.imshow('text', img)
            cv2.waitKey()
            cv2.destroyWindow('text')
