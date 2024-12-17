# Rail Riders

This project automates Rail Riders, where the player must swipe from the bottom to the top of the lower screen as quickly as possible. The automation is achieved using Python and several libraries to simulate mouse movements and keyboard inputs.

### Demo Video
[![Mario Party DS Rail Riders Demo](https://www.youtube.com/embed/IVxGimlQ3eY?si=K6AjJIrMrVddgGez)]

## Features

- **Automated Swiping**: The script simulates rapid swiping on the emulator's touch screen.
- **Adjustable Timing**: The script allows for experimentation with different time delays to optimize performance.

## Techniques Used

The automation leverages several techniques:

1. **Mouse Movement Simulation**: Using the `pyautogui` library to simulate mouse movements and clicks on the emulator window.
2. **Keyboard Input Detection**: Utilizing the `keyboard` library to listen for an emergency exit key.
3. **Window Management**: Accessing and activating the emulator window using `pygetwindow` to ensure the automation interacts with the correct application.
4. **Timing Adjustments**: Experimenting with different delays between actions to find optimal swipe speed and responsiveness.
