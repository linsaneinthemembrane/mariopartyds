import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from datetime import datetime
import time

def detect_bright_stars(frame):
    # Set minimum RGB threshold (adjust these values as needed)
    min_rgb = np.array([250, 250, 250])
    
    # Create mask for pixels brighter than threshold
    bright_mask = np.all(frame >= min_rgb, axis=-1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask.astype(np.uint8), connectivity=8)
    
    # Get star positions
    star_positions = []
    for i in range(1, num_labels):
        if 4 < stats[i, cv2.CC_STAT_AREA] < 50:
            x = int(centroids[i][0])
            y = int(centroids[i][1])
            star_positions.append((x, y))
    
    return star_positions, bright_mask.astype(np.uint8) * 255

def click_position(x, y, left, bottom_half_top):
    pyautogui.mouseDown(left + x, bottom_half_top + y)
    time.sleep(0.03)  
    pyautogui.mouseUp()
    time.sleep(0.02)  # Best amount of time I could find

def main():
    start_time = datetime.now()
    timeout_seconds = 70
    
    # Initialize emulator window
    emulator_window = gw.getWindowsWithTitle("[60/60] melonDS 1.0 RC")[0]
    emulator_window.activate()
    
    # Get window dimensions
    left, top, width, height = emulator_window.left, emulator_window.top, emulator_window.width, emulator_window.height
    bottom_half_top = top + height // 2
    bottom_half_height = height // 2
    
    # Create display window
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", width, bottom_half_height)
    
    while True:
        # Automatically closes the program after 45 seconds have passed
        if (datetime.now() - start_time).total_seconds() > timeout_seconds:
            print("45 second timeout reached")
            break
        
        # Shows what the computer is looking at with the mask
        screenshot = pyautogui.screenshot(region=(left, bottom_half_top, width, bottom_half_height))
        current_frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Detect stars
        star_positions, white_mask = detect_bright_stars(current_frame)
        
        # Show mask
        cv2.imshow("Mask", white_mask)
        
        # Click detected stars
        for x, y in star_positions:
            click_position(x, y, left, bottom_half_top)
        
        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    main()
