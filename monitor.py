import screeninfo
import threading
from pynput import mouse, keyboard
import time
from config import MONITOR

def monitor_user_action(target_col, target_row):
    """
    Monitors user interactions within a target grid area to confirm task completion through timed action validation.
    
    This method divides the screen into a 3x3 grid and monitors for mouse clicks and keyboard presses within 
    a specified target square. When a click is detected in the target area, a timer is started to validate that 
    the user has engaged with the correct UI element. The timer resets with each subsequent keyboard press, 
    allowing the system to track ongoing user interaction. The action is considered complete when the timer 
    expires without interruption, confirming the user has successfully interacted with the designated area 
    and providing feedback to the server about task progression.
    
    Args:
        target_col (int): The column number (1-3) of the target grid square.
        target_row (int): The row number (1-3) of the target grid square.
    
    Returns:
        bool: True if the user successfully completed the timed action (timer expired without interruption), 
            False otherwise.
    """
    action_completed = False  # Флаг завершения действия
    timer_started = False     # Таймер запускается после клика

    # Получаем размеры экрана
    screen = screeninfo.get_monitors()[0]
    SCREEN_WIDTH = screen.width
    SCREEN_HEIGHT = screen.height

    # Рассчитываем размеры каждого квадрата в сетке 3x3
    square_width = SCREEN_WIDTH // MONITOR['GRID_DIVISIONS']
    square_height = SCREEN_HEIGHT // MONITOR['GRID_DIVISIONS']

    # Определяем границы целевого квадрата
    x_start = (target_col - 1) * square_width
    y_start = (target_row - 1) * square_height
    x_end = target_col * square_width
    y_end = target_row * square_height

    # Таймер и блокировки
    timer = None
    timer_lock = threading.Lock()

    # Функция проверки, находится ли касание в целевой области
    def is_in_target_area(x, y):
        return x_start <= x < x_end and y_start <= y < y_end

    # Обработчик нажатия мыши
    def on_click(x, y, button, pressed):
        nonlocal timer_started
        if pressed and is_in_target_area(x, y):  # Если нажата кнопка мыши и касание в целевой области
            print("Клик в целевой области! Таймер запущен на 2 секунды.")
            timer_started = True
            start_timer()

    # Обработчик нажатия клавиш для сброса таймера
    def on_key_press(key):
        if timer_started:
            print("Нажата клавиша. Таймер сброшен.")
            reset_timer()

    # Функция, запускающая таймер после последнего нажатия клавиши
    def start_timer():
        nonlocal timer
        with timer_lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(MONITOR['TIMER_DURATION'], timer_expired)
            timer.start()

    # Функция для сброса и перезапуска таймера
    def reset_timer():
        start_timer()

    # Функция, вызываемая по истечении таймера
    def timer_expired():
        nonlocal action_completed
        action_completed = True  # Пользователь выполнил действие

    # Запуск слушателей для мыши и клавиатуры
    with mouse.Listener(on_click=on_click) as mouse_listener, keyboard.Listener(on_press=on_key_press) as key_listener:
        while not action_completed:
            time.sleep(0.1)
        # Остановка слушателей после выполнения действия
        mouse_listener.stop()
        key_listener.stop()
        with timer_lock:
            if timer is not None:
                timer.cancel()

    return action_completed