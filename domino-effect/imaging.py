import win32gui
import win32con
from mss import mss
import numpy as np

class GameCapture:
    def __init__(self):
        self.sct = mss()
        self.window_title = "[60/60] melonDS 1.0 RC" #[60/60] 

    def bring_window_to_foreground(self, hwnd):
        try:
            # Bring the window to the foreground
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            print(f"Error bringing window to foreground: {e}")

    def get_window_geometry(self):
        hwnd = win32gui.FindWindow(None, self.window_title)

        if hwnd:
            # Bring emulator to foreground
            self.bring_window_to_foreground(hwnd)

            # Get the current window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0]
            y = rect[1]
            width = rect[2] - x
            height = rect[3] - y

            # Debug: Print window geometry
            return {"left": x, "top": y, "width": width, "height": height}

        # Fallback to capture the primary monitor
        print("Emulator window not found. Capturing full monitor.")
        return self.sct.monitors[1]

    def capture_frame(self):
        geometry = self.get_window_geometry()
        screenshot = self.sct.grab(geometry)

        # Convert to a numpy array
        frame = np.array(screenshot)
        return frame
