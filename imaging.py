import win32gui
import win32ui
import win32con
import numpy as np
from multiprocessing import Process, Queue
import time

class ImageProcessor:
    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f'Window not found: {window_name}')
        self.capture_interval = 0.05  # 50ms interval, adjust if needed
        self.processing_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        self.running = True

    def capture_window(self):
        # Get window dimensions
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        w = right - left
        h = bot - top

        # Create device context
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)

        # Convert to numpy array
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        # Clean up resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        return img

    def capture_images(self):
        while self.running:
            screenshot = self.capture_window()
            if not self.processing_queue.full():
                self.processing_queue.put(screenshot)
            time.sleep(self.capture_interval)

    def process_images(self):
        while self.running:
            if not self.processing_queue.empty():
                screenshot = self.processing_queue.get()
                self.result_queue.put(screenshot)

    def run(self):
        capture_process = Process(target=self.capture_images)
        process_process = Process(target=self.process_images)

        capture_process.start()
        process_process.start()

        return self.result_queue

    def stop(self):
        self.running = False

if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor("melonDS emulator")  # Replace with your emulator window title
    result_queue = processor.run()

    try:
        while True:
            if not result_queue.empty():
                frame = result_queue.get()
                # You can manually inspect frame data here
            time.sleep(0.1)
    except KeyboardInterrupt:
        processor.stop()
        print("Image processing stopped.")