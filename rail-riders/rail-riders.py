import keyboard as key
import pyautogui as gui
import time
import pygetwindow as gw

def ride_rails():
    # Set the delay for PyAutoGUI actions to swipe faster
    gui.PAUSE = 0.01
    
    # Activate the emulator window
    emulator_window = gw.getWindowsWithTitle("[60/60] melonDS 1.0 RC")[0]
    emulator_window.activate()

    # Get window position and dimensions
    left, top, width, height = emulator_window.left, emulator_window.top, emulator_window.width, emulator_window.height

    # Define swipe start and end positions
    start_drag = (left + width / 2, top + height * (15 / 16))
    end_drag = (left + width / 2, top + height * (9 / 16))  # Optimal values from experimentation

    # Delay to allow time for the game window to load
    time.sleep(.5)

    start_time = time.time()
    
    while time.time() - start_time <= 20:  # Run for 20 seconds
        # press q to break
        if key.is_pressed('q'):
            break
        
        # Perform the swipe action
        gui.moveTo(start_drag)
        gui.mouseDown()
        gui.moveTo(end_drag, duration=0.1)  # Swipe approximately 10 times per second
        gui.mouseUp()

    # Reset PyAutoGUI delay to default value
    gui.PAUSE = 0.1

if __name__ == "__main__":
    ride_rails()
