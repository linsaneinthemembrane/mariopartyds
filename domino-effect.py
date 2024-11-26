import cv2
import numpy as np
from imaging import ImageProcessor

def test_domino_pixels():
    # Initialize the image processor with your emulator window title
    processor = ImageProcessor("melonDS emulator")  # Adjust window name as needed
    result_queue = processor.run()

    try:
        while True:
            if not result_queue.empty():
                # Get the latest frame
                frame = result_queue.get()
                
                # Display the full frame
                cv2.imshow("Full Game Window", frame)
                
                # Extract and display the B button region
                b_button_region = frame[750:850, 800:900]  # Adjust coordinates as needed
                cv2.imshow("B Button Region", b_button_region)
                
                # Get pixel values at mouse click
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        # Print RGB values at clicked position
                        color = frame[y, x]
                        print(f"Pixel at ({x}, {y}): RGB = {color[:3]}")
                
                cv2.setMouseCallback("Full Game Window", mouse_callback)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_domino_pixels()