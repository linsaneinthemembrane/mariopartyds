import numpy as np
import cv2
from imaging import GameCapture
from collections import deque

class TraceCadets:
    def __init__(self):
        self.capture = GameCapture()
        
        # RGB values from the hex #289cfc (40, 156, 252)
        self.blue_lower = np.array([100, 200, 220])
        self.blue_upper = np.array([140, 255, 255])
        
        # Parameters for shape detection
        self.min_area = 100
        self.max_area = 5000
        
        # Initialize shape queue
        self.shape_queue = deque(maxlen=5)
        
        # Grid parameters
        self.grid_region = {
            'top': 925,    
            'left': 60,   
            'width': 500,  
            'height': 325
        }
        
        # Scaling and grid parameters
        self.scale_factor = 6
        self.grid_cols = 6
        self.grid_rows = 4
        
        # Calculate scaled dimensions
        self.scaled_width = int(self.grid_region['width'] * self.scale_factor)
        self.scaled_height = int(self.grid_region['height'] * self.scale_factor)
        self.cell_width = self.scaled_width // self.grid_cols
        self.cell_height = self.scaled_height // self.grid_rows

    def detect_shapes(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        inverted_mask = cv2.bitwise_not(mask)
        inverted_mask_color = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)
        
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area and area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                shapes.append({
                    'contour': contour,
                    'position': (x, y),
                    'size': (w, h),
                    'area': area
                })
        
        if shapes:
            shapes.sort(key=lambda s: (-s['area'], s['position'][1], s['position'][0]))
            for i, shape in enumerate(shapes):
                color = (0, 255, 0) if i == 0 else (0, 0, 0)
                cv2.drawContours(inverted_mask_color, [shape['contour']], -1, color, 2)
        
        cv2.imshow('Original Mask', mask)
        cv2.imshow('Inverted Mask', inverted_mask_color)
        
        return shapes

    def scale_shape(self, shape):
        # Use fixed scale of 20 as determined by testing
        shape_scale = 19.5
        
        contour_float = shape['contour'].astype(np.float32)
        scaled_contour = contour_float * shape_scale
        
        # Ensure shape starts at (0,0) before applying offset
        min_x = scaled_contour[:, :, 0].min()
        min_y = scaled_contour[:, :, 1].min()
        scaled_contour[:, :, 0] -= min_x
        scaled_contour[:, :, 1] -= min_y
        
        # Add small offset from grid edge
        start_x, start_y = 30, 30  # Starting point with slight padding
        scaled_contour[:, :, 0] += start_x
        scaled_contour[:, :, 1] += start_y
        
        return scaled_contour.astype(np.int32)

    def draw_scaled_shape_on_grid(self, grid_image, shape):
        scaled_grid = cv2.resize(grid_image, (self.scaled_width, self.scaled_height))
        
        # Draw grid lines
        for i in range(self.grid_cols + 1):
            x = int(i * self.cell_width)
            cv2.line(scaled_grid, (x, 0), (x, self.scaled_height), (0, 0, 0), 2)
        
        for i in range(self.grid_rows + 1):
            y = int(i * self.cell_height)
            cv2.line(scaled_grid, (0, y), (self.scaled_width, y), (0, 0, 0), 2)
        
        # Draw scaled shape
        if shape is not None:
            scaled_contour = self.scale_shape(shape)
            cv2.drawContours(scaled_grid, [scaled_contour], -1, (0, 255, 0), 3)
        
        return scaled_grid

    def get_drawing_grid(self, frame):
        grid = frame[
            self.grid_region['top']:self.grid_region['top'] + self.grid_region['height'],
            self.grid_region['left']:self.grid_region['left'] + self.grid_region['width']
        ]
        return grid

    def run(self):
        cv2.namedWindow('Trace Cadets Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Original Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Inverted Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Drawing Grid', cv2.WINDOW_NORMAL)
        
        while True:
            frame = np.array(self.capture.capture_frame())
            
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            shapes = self.detect_shapes(frame)
            debug_frame = frame.copy()
            drawing_grid = self.get_drawing_grid(frame)
            
            if shapes:
                for i, shape in enumerate(shapes):
                    color = (0, 255, 0) if i == 0 else (0, 0, 0)
                    cv2.drawContours(debug_frame, [shape['contour']], -1, color, 2)
                
                scaled_grid = self.draw_scaled_shape_on_grid(drawing_grid, shapes[0])
                cv2.imshow('Drawing Grid', scaled_grid)
            else:
                cv2.imshow('Drawing Grid', drawing_grid)
            
            cv2.imshow('Trace Cadets Detection', debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = TraceCadets()
    game.run()

