# Star Catchers

This project uses computer vision techniques to detect and automatically click on stars as they appear in the Star Catchers minigame. The game requires quick reactions to click white stars before other players can, making it an ideal candidate for automation.

### Demo Video
[![Mario Party DS Star Catchers Demo](https://img.youtube.com/vi/x3u6SKqFIg8/maxresdefault.jpg)](https://youtu.be/x3u6SKqFIg8)
Here is a quick demo; the program beat all Expert CPUs, leaving each with zero stars, something I dreamed about doing as a kid.

## Detection Method

- Initially attempted RGB averaging to detect stars
- Evolved to using binary masking for pure white pixels
- Implemented connected components analysis to identify star shapes
- Fine-tuned detection thresholds to catch stars in their first frame of appearance

## Optimization Process
- Started with full frame processing
- Reduced to bottom screen only
- Implemented binary mask visualization for debugging3
- Adjusted click timing and detection parameters

## Challenges Overcome

### Star Detection
- Initial difficulty distinguishing new stars from claimed ones2
- Stars would blink briefly white before changing color
- Multiple stars appearing simultaneously
- False positives from similar colored pixels

### Performance
- Needed to balance detection speed with accuracy
- Click timing required fine-tuning
- Frame capture and processing optimization
- Window management and coordinate mapping

## Key Features
- Real-time star detection using OpenCV
- Automated mouse control with PyAutoGUI
- Visual mask feedback for monitoring

### License
MIT License
