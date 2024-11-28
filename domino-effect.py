import numpy as np
import time
import cv2
from imaging import GameCapture
import keyboard

class DominoEffect:
    def __init__(self):
        self.capture = GameCapture()
        # may have to adjust these to your button regions
        self.button_regions = {
            'first': {'top': 851, 'left': 554, 'width': 45, 'height': 52},
            'second': {'top': 890, 'left': 640, 'width': 45, 'height': 52}
        }
        
        self.button_colors = {
            'A': [
                np.array([191, 213, 225]),
                np.array([226, 250, 254])
            ],
            'B': [
                np.array([182, 207, 220]),
                np.array([224, 250, 254])
            ],
            'X': [np.array([190, 214, 226])],
            'Y': [np.array([198, 218, 228])]
        }
                
        self.initialize_windows()
    
    def initialize_windows(self):
        cv2.namedWindow('Full Window', cv2.WINDOW_NORMAL)
        cv2.namedWindow('First Button Region', cv2.WINDOW_NORMAL)
        self.current_first_region = None
        cv2.setMouseCallback('First Button Region', self.first_region_click)
    
    def get_button_region(self, frame, region):
        try:
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            return frame[
                region['top']:region['top'] + region['height'],
                region['left']:region['left'] + region['width']
            ]
        except Exception as e:
            print(f"Error getting region: {e}")
            return np.zeros((52, 45, 3), dtype=np.uint8)
    
    def get_average_color(self, region):
        return np.mean(region, axis=(0,1)).astype(int)
    
    def find_button_by_rgb(self, detected_rgb):
        def rgb_distance(rgb1, rgb2):
            return np.linalg.norm(rgb1 - rgb2)
        
        min_distance = float('inf')
        closest_button = None
        threshold = 15
        
        for button, rgb_list in self.button_colors.items():
            for rgb in rgb_list:
                distance = rgb_distance(detected_rgb, rgb)
                if distance < min_distance:
                    min_distance = distance
                    closest_button = button
        
        return closest_button if min_distance < threshold else None
    
    def first_region_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and hasattr(self, 'current_first_region'):
            color = self.get_average_color(self.current_first_region)
            button = self.find_button_by_rgb(color)
            print(f"\nFirst Region - RGB: {color}, Detected Button: {button}")
    
    def press_button(self, button):
        if button:
            print(f"\n[DEBUG] Pressing button: {button}")
            try:
                if button == 'A':
                    keyboard.press('a')
                    time.sleep(0.05)
                    keyboard.release('a')
                elif button == 'B':
                    keyboard.press('b')
                    time.sleep(0.05)
                    keyboard.release('b')
                elif button == 'X':
                    keyboard.press('x')
                    time.sleep(0.05)
                    keyboard.release('x')
                elif button == 'Y':
                    keyboard.press('y')
                    time.sleep(0.05)
                    keyboard.release('y')
                print(f"[DEBUG] Button {button} successfully pressed.")
            except Exception as e:
                print(f"[ERROR] Failed to press button {button}: {e}")
    def process_buttons(self, first_button, second_button):
        if first_button:
            return first_button
        return None

    def run(self):
        last_press_time = 0
        while True:
            current_time = time.time()
            frame = np.array(self.capture.capture_frame())
            
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Get both regions
            first_region = self.get_button_region(frame, self.button_regions['first'])
            second_region = self.get_button_region(frame, self.button_regions['second'])
            
            # Detect buttons in both regions
            first_button = self.find_button_by_rgb(self.get_average_color(first_region))
            second_button = self.find_button_by_rgb(self.get_average_color(second_region))
            
            # Process buttons using stack
            button_to_press = self.process_buttons(first_button, second_button)
            
            if current_time - last_press_time >= 0.07:
                self.press_button(button_to_press)
                last_press_time = current_time
            
            # Display windows
            cv2.imshow('Full Window', frame)
            cv2.imshow('First Button Region', first_region)
            cv2.imshow('Second Button Region', second_region)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(1/30)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = DominoEffect()
    game.run()
