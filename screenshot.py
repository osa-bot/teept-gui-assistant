import pyautogui
from PIL import Image, ImageDraw, ImageFont
import io
from config import GRID

def take_screenshot():
    """
    Captures a screenshot and overlays a 3x3 grid with labeled regions for task guidance.
    
    This method captures the current screen and divides it into a 3x3 grid to provide visual 
    guidance for user interactions. Each grid cell is labeled with its coordinates, enabling 
    users to understand which screen region corresponds to specific actionable areas during 
    task execution. The grid serves as a reference system for mapping server-provided 
    instructions to actual screen locations.
    
    Args:
        None
    
    Returns:
        tuple: A tuple containing:
            - screenshot (PIL.Image): PIL Image object of the screenshot with grid overlay and coordinate labels.
            - buffered (io.BytesIO): BytesIO buffer containing the JPEG-encoded screenshot for transmission or storage.
    """
    # Сделать скриншот
    screenshot = pyautogui.screenshot()

    # Создаем объект для рисования
    draw = ImageDraw.Draw(screenshot)

    # Получаем размеры изображения
    width, height = screenshot.size

    # Вычисляем позиции для линий сетки
    third_width = width // GRID['COLUMNS']
    third_height = height // GRID['ROWS']

    # Рисуем вертикальные линии
    for i in range(1, 3):
        x = i * third_width
        draw.line([(x, 0), (x, height)], fill=GRID['LINE_COLOR'], width=GRID['LINE_WIDTH'])

    # Рисуем горизонтальные линии
    for i in range(1, 3):
        y = i * third_height
        draw.line([(0, y), (width, y)], fill=GRID['LINE_COLOR'], width=GRID['LINE_WIDTH'])

    # Добавляем номера к каждой области
    font_size = 36  # Размер шрифта
    try:
        # Попытка загрузить шрифт Arial
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Если не удалось загрузить Arial, используем шрифт по умолчанию
        font = ImageFont.load_default()

    for row in range(3):
        for col in range(3):
            x = col * third_width + third_width // 2
            y = row * third_height + third_height // 2
            text = f"X: {row + 1} Y: {col + 1}"
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x - text_width // 2
            text_y = y - text_height // 2
            draw.text((text_x, text_y), text, fill='red', font=font)

    # Сохранить измененное изображение в буфер в формате JPEG
    buffered = io.BytesIO()
    screenshot.save(buffered, format="JPEG")
    buffered.seek(0)

    return screenshot, buffered
    
def display_sent_image(screenshot):
    """
    Displays a screenshot for visual feedback during task execution.
    
    This method renders a screenshot in a separate window to provide users with
    real-time visual confirmation of the current state of the desktop task being
    executed. The window title identifies the image as a sent/captured screenshot,
    helping users track the progression of automated actions and UI element detection.
    
    Args:
        screenshot: The image object containing the captured screenshot to be displayed.
    
    Returns:
        None
    """
    # Отображаем изображение в отдельном окне
    screenshot.show(title="Отправленное изображение")
    