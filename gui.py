from PyQt5 import QtWidgets, QtGui, QtCore
import ctypes
import win32con
import threading
import time
from pynput import mouse, keyboard
import screeninfo
import sys
from config import OVERLAY

class Overlay(QtWidgets.QWidget):
    """
    Overlay window for displaying and highlighting elements with visual annotations.
    
        This class creates a frameless, always-on-top overlay window that displays full-screen
        with transparent background. It renders visual highlights for search areas and detected
        elements with their associated metadata such as similarity scores and text content.
        Windows-specific window properties are configured to enable layered window rendering
        for proper transparency handling.
    
        Attributes:
            elements: List of dictionaries containing element information with bounding rectangles,
                text content, similarity scores, and search area flags.
    
        Methods:
            - __init__: Initializes the overlay window with element data and window properties.
            - paintEvent: Renders visual highlights and annotations for elements on the overlay.
    """

    def __init__(self, elements):
        """
        Initializes an overlay window for displaying detected UI elements with visual highlights and guidance.
        
        This constructor sets up a frameless, always-on-top window that displays full-screen
        with transparent background. It configures Windows-specific window properties to enable
        layered window rendering for proper transparency handling. The overlay serves as the visual
        feedback mechanism that guides users through task execution by highlighting and annotating
        detected UI components on the screen.
        
        Args:
            elements: A list of dictionaries containing detected UI element information, where each
                dictionary includes 'rect' (bounding rectangle coordinates), 'text_content' (OCR or 
                extracted text from the element), 'similarity' (confidence score for element matching),
                and 'is_search_area' (boolean flag indicating if the element is a designated search region).
        
        Returns:
            None
        
        Attributes:
            elements: List of dictionaries with element data including rectangles, text content,
                similarity scores, and search area flags used for rendering visual overlays.
        """
        super().__init__()
        self.elements = elements  # Список словарей с 'rect', 'text_content', 'similarity', 'is_search_area'
        self.setWindowTitle('Overlay')
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.showFullScreen()

        hwnd = self.winId().__int__()
        extendedStyle = ctypes.windll.user32.GetWindowLongW(hwnd, win32con.GWL_EXSTYLE)
        extendedStyle |= win32con.WS_EX_LAYERED
        ctypes.windll.user32.SetWindowLongW(hwnd, win32con.GWL_EXSTYLE, extendedStyle)

    def paintEvent(self, event):
        """
        Renders visual overlays for detected UI elements and search regions on the screen.
        
        This method paints interactive visual feedback to guide users through task automation by drawing
        outlined rectangles for both search areas and detected UI elements. For each detected element,
        it displays associated metadata including similarity scores and extracted text content, enabling
        users to verify that the correct UI components have been identified before interaction.
        
        The method uses different visual styles to distinguish between search regions (dashed lines) and
        found elements (solid red lines), providing clear visual hierarchy for the user's guidance.
        
        Args:
            event: The paint event that triggered this method.
        
        Returns:
            None
        """
        painter = QtGui.QPainter(self)
        font = QtGui.QFont()
        font.setPointSize(12)
        painter.setFont(font)
        # Рисование прямоугольников и текста
        for elem in self.elements:
            rect = elem['rect']
            if elem.get('is_search_area', False):
                # Устанавливаем кисть для области поиска
                painter.setPen(QtGui.QPen(QtGui.QColor(*OVERLAY['SEARCH_AREA_COLOR']), 
                         OVERLAY['LINE_WIDTH']['SEARCH'], 
                         OVERLAY['DASH_LINE']))

                painter.drawRect(rect)
            else:
                # Устанавливаем кисть для найденных элементов
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 3))
                painter.drawRect(rect)
                # Формируем текст для отображения
                texts = []
                if 'similarity' in elem:
                    texts.append(f"Сходство: {elem['similarity']:.2f}")
                if 'text_content' in elem and elem['text_content']:
                    texts.append(f"Текст: {elem['text_content']}")
                if texts:
                    text_str = '\n'.join(texts)
                    # Отрисовка текста рядом с прямоугольником
                    text_rect = QtCore.QRect(rect.right() + 5, rect.top(), 200, 50)
                    painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, text_str)

app = QtWidgets.QApplication(sys.argv)