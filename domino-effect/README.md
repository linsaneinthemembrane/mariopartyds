# Mario Party DS Domino Effect Bot
A Python-based bot that automatically plays the Domino Effect minigame from Mario Party DS by detecting and responding to on-screen button prompts.

### Demo Video
[![Mario Party DS Domino Effect Bot Demo](https://img.youtube.com/vi/QWJFMsfa0ag/maxresdefault.jpg)](https://www.youtube.com/watch?v=QWJFMsfa0ag)

## Overview
This bot uses computer vision to detect button prompts (A, B, X, Y) appearing in the bottom-middle of the screen and automatically presses the corresponding keys to play the Domino Effect minigame.
### Features
* Real-time window capture of the melonDS emulator
* RGB-based button prompt detection
* Automated key pressing
* Debug visualization windows
* Support for multiple button detection regions
* Stack-based sequence tracking
## Development Process & Challenges
### Initial Approach
* Started with single-window detection focusing on one button prompt
* Used RGB color matching to identify button types
* Implemented basic key pressing functionality
### Stack Implementation Motivation
The decision to use a stack data structure was inspired by:
* Recent LeetCode problem-solving experience
* The natural LIFO (Last In, First Out) pattern of the game's button sequence
* Buttons appear right-to-left in the game, matching stack's natural order
* Efficient O(1) operations for adding and removing buttons
## Key Challenges Overcome
1. Window Detection
* Initially captured full desktop instead of emulator window
* Solved by implementing proper window title detection and focusing
2. Button Recognition
* Different RGB values for same buttons in different screen positions
* Solved by storing multiple RGB values per button type
3. Input Registration
* Game wasn't recognizing keyboard inputs initially
* Resolved by switching from arrow keys to direct button mapping (A, B, X, Y)
* Added proper timing between button presses (0.07s intervals)
4. Sequence Management
* Evolved from single button detection to stack-based sequence tracking
* Implemented deque with maxlen for efficient button history management
* Stack perfectly matches the right-to-left button appearance pattern
### Technical Details
* Uses OpenCV for image processing
* Keyboard library for input simulation
* Win32GUI for window management
* NumPy for array operations
* Collections.deque for stack implementation
### Requirements
* Python 3.12
* OpenCV
* Keyboard module
* NumPy
* MSS for screen capture
* Win32GUI
### Usage
* Start melonDS emulator with Mario Party DS
* Navigate to Domino Effect minigame
* Run the script
* Press 'q' to quit
### Installation
```pip install opencv-python numpy keyboard mss pywin32```
### License
MIT License
