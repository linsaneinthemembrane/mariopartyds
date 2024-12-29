import cv2
import numpy as np
import pyautogui as gui
import pygetwindow as gw
import keyboard as key
import time
from typing import List, Dict, Tuple

class ShapeProperties:
    def __init__(self, contour):
        self.contour = contour
        self.width = None
        self.height = None
        self.orientation = None
        self.aspect_ratio = None
        self.area = None
        self.grid_cells = None
        
        # Calculate basic properties
        x, y, w, h = cv2.boundingRect(contour)
        self.width = w
        self.height = h
        self.aspect_ratio = w / h if h != 0 else 0
        self.area = cv2.contourArea(contour)
        
        # Determine orientation
        if self.aspect_ratio > 1.2:
            self.orientation = 'horizontal'
        elif self.aspect_ratio < 0.8:
            self.orientation = 'vertical'
        else:
            self.orientation = 'square'
            
        # Set grid cells
        self.grid_cells = (round(w / 100), round(h / 100))

class ShapeTracer:
    def __init__(self):
        self.window_name = "[60/60] melonDS 1.0 RC"
        self.grid_width = 3
        self.grid_height = 2
        self.grid_cell_size = 100
        self.grid_offset_x = 205
        self.grid_offset_y = 225
        self.cell_size = 165
        
        cv2.namedWindow('Recognized Shapes', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tracing Progress', cv2.WINDOW_NORMAL)
        
        self.blue_lower = np.array([100, 150, 150])
        self.blue_upper = np.array([140, 255, 255])

    def get_emulator_window(self):
        try:
            emulator = gw.getWindowsWithTitle(self.window_name)[0]
            emulator.activate()
            return emulator
        except IndexError:
            raise Exception("Emulator window not found")

    def capture_screen(self, emulator) -> Tuple[np.ndarray, np.ndarray]:
        screenshot = np.array(gui.screenshot(region=(
            emulator.left, emulator.top,
            emulator.width, emulator.height
        )))
        
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        icon_width = emulator.width // 8
        top_screen = screenshot[:emulator.height//2, icon_width:]
        bottom_screen = screenshot[emulator.height//2:, :]
        
        return top_screen, bottom_screen

    def process_shapes(self, image: np.ndarray) -> List[Dict]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        kernel = np.ones((3,3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(
            blue_mask, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        shapes = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            
            for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
                if hier[3] != -1:
                    area = cv2.contourArea(cnt)
                    if area < 100:
                        continue
                    
                    shape_props = ShapeProperties(cnt)
                    epsilon = 0.03 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    shapes.append({
                        'contour': cnt,
                        'properties': shape_props,
                        'approx': approx,
                        'area': area
                    })
        
        return sorted(shapes, key=lambda x: x['area'], reverse=True)

    def scale_contour(self, contour: np.ndarray, scale_factor: float) -> np.ndarray:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # Ensure contour has correct shape (N,1,2)
            contour = contour.reshape(-1, 1, 2)
            
            scaled_contour = contour.astype(np.float32)
            scaled_contour -= [cx, cy]
            scaled_contour *= scale_factor
            scaled_contour += [cx, cy]
            
            # Force shape closure
            first_point = scaled_contour[0].reshape(1, 1, 2)
            scaled_contour = np.concatenate([scaled_contour, first_point], axis=0)
            
            return scaled_contour.astype(np.int32)
        return contour


    def draw_grid(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        
        # Draw vertical lines
        for i in range(self.grid_width + 1):
            x = self.grid_offset_x + (i * self.cell_size)
            if 0 <= x < w:
                cv2.line(image, (x, self.grid_offset_y), 
                        (x, self.grid_offset_y + self.grid_height * self.cell_size), 
                        (255, 255, 255), 1)
        
        # Draw horizontal lines
        for i in range(self.grid_height + 1):
            y = self.grid_offset_y + (i * self.cell_size)
            if 0 <= y < h:
                cv2.line(image, (self.grid_offset_x, y), 
                        (self.grid_offset_x + self.grid_width * self.cell_size, y), 
                        (255, 255, 255), 1)
        
        return image

    def draw_recognized_shapes(self, shapes: List[Dict], image: np.ndarray) -> np.ndarray:
        debug_image = image.copy()
        for shape in shapes:
            cv2.drawContours(debug_image, [shape['contour']], -1, (0, 255, 0), 2)
            props = shape['properties']
            info_text = f"{props.orientation} {props.grid_cells}"
            cv2.putText(debug_image, info_text, 
                       tuple(shape['contour'][0][0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return debug_image

    def draw_trace_preview(self, shape: Dict, bottom_screen: np.ndarray, 
                          offset_x: int = 0, 
                          offset_y: int = 0) -> np.ndarray:
        preview = bottom_screen.copy()
        props = shape['properties']
        
        # Draw grid only once
        preview = self.draw_grid(preview)
        
        scale_factor = 4.5
        x, y, w, h = cv2.boundingRect(shape['contour'])
        scaled_contour = self.scale_contour(shape['contour'], scale_factor)
        
        # Use grid offsets for positioning
        base_x = self.grid_offset_x
        base_y = self.grid_offset_y
        
        M = np.float32([[1, 0, base_x - x * scale_factor],
                        [0, 1, base_y - y * scale_factor]])
        
        translated_contour = scaled_contour + [M[0][2], M[1][2]]
        
        cv2.drawContours(preview, [translated_contour.astype(np.int32)], -1, (0, 255, 0), 2)
        
        # Display parameters
        cv2.putText(preview, f"Scale: {scale_factor:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(preview, f"Offset X: {offset_x}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(preview, f"Offset Y: {offset_y}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(preview, f"Grid: {props.grid_cells}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(preview, f"Orientation: {props.orientation}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return preview

    def calculate_grid_position(self, emulator, shape: Dict, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert grid coordinates to emulator window coordinates"""
        window_x = emulator.left
        window_y = emulator.top + (emulator.height // 2)
        
        props = shape['properties']
        # All shapes should start from rightmost position
        grid_x = 1  # Always start from rightmost column
        grid_y = 0  # Always start from top row
        
        # Calculate screen coordinates based on grid position
        screen_x = window_x + self.grid_offset_x + (grid_x * self.cell_size)
        screen_y = window_y + self.grid_offset_y
        
        return (screen_x, screen_y)




    def add_intermediate_points(self, trace_points):
        """Add intermediate points with slight random offsets for more natural movement"""
        new_trace_points = []
        for i in range(len(trace_points) - 1):
            start = trace_points[i]
            end = trace_points[i + 1]
            
            # Add the start point
            new_trace_points.append(start)
            
            if abs(end[0] - start[0]) > 0 and abs(end[1] - start[1]) > 0:
                # Single intermediate point for diagonal
                intermediate_x = start[0] + (end[0] - start[0]) // 2
                intermediate_y = start[1] + (end[1] - start[1]) // 2
                new_trace_points.append((intermediate_x, intermediate_y))
        
        # Add the final point
        new_trace_points.append(trace_points[-1])
        return new_trace_points


    def generate_trace_points(self, emulator, shape: Dict, grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
        contour = shape['contour']
        
        # Scale contour before calculating position
        scaled_contour = self.scale_contour(contour, 3.22)
        
        # Get start position in emulator window
        start_x, start_y = self.calculate_grid_position(emulator, shape, grid_x, grid_y)
        
        # Generate points relative to emulator window
        trace_points = []
        for point in scaled_contour:
            # Adjust point calculation to match preview
            x = start_x + 15 + (point[0][0] - scaled_contour[0][0][0])
            y = start_y + 15 + (point[0][1] - scaled_contour[0][0][1])
            trace_points.append((x, y))
        
        return trace_points


    def trace_shape(self, emulator, shape: Dict, grid_x: int, grid_y: int):
        """Execute the tracing movement with more natural motion"""
        trace_points = self.generate_trace_points(emulator, shape, grid_x, grid_y)
        
        # Move to start position without clicking
        gui.moveTo(trace_points[0][0], trace_points[0][1])        
        # Start drawing
        gui.mouseDown()
        
        # Trace with slower, more natural movement
        for point in trace_points[1:]:
            gui.moveTo(point[0], point[1], duration=0.1)  # Slower movement
        
        # Ensure we return to start point
        gui.moveTo(trace_points[0][0], trace_points[0][1], duration=0.1)
        time.sleep(0.1)  # Pause before releasing
        
        gui.mouseUp()



    def start(self):
        emulator = self.get_emulator_window()
        start_time = time.time()
        
        offset_x = 0
        offset_y = 0
        position_step = 5
        
        while True:
            """ if time.time() - start_time > 20:
                print("20 seconds elapsed - stopping program")
                break """
                
            if key.is_pressed('up'):
                offset_y -= position_step
            if key.is_pressed('down'):
                offset_y += position_step
            if key.is_pressed('left'):
                offset_x -= position_step
            if key.is_pressed('right'):
                offset_x += position_step
                
            top_screen, bottom_screen = self.capture_screen(emulator)
            available_shapes = self.process_shapes(top_screen)
            
            if key.is_pressed('t'):
                if available_shapes:
                    shape = available_shapes[0]
                    grid_x = int(shape['properties'].grid_cells[0])
                    grid_y = int(shape['properties'].grid_cells[1])
                    self.trace_shape(emulator, shape, grid_x, grid_y)
                    time.sleep(0.5)

            debug_view = self.draw_recognized_shapes(available_shapes, top_screen)
            cv2.imshow('Recognized Shapes', debug_view)
            
            if available_shapes:
                preview = self.draw_trace_preview(
                    available_shapes[0], 
                    bottom_screen,
                    offset_x=offset_x,
                    offset_y=offset_y
                )
                cv2.imshow('Tracing Progress', preview)
            else:
                cv2.imshow('Tracing Progress', bottom_screen)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or key.is_pressed('q'):
                if available_shapes:
                    props = available_shapes[0]['properties']
                    print("\nFinal Parameters:")
                    print(f"Orientation: {props.orientation}")
                    print(f"Grid Cells: {props.grid_cells}")
                    print(f"Scale Factor: 3.20")
                    print(f"X Offset: {offset_x}")
                    print(f"Y Offset: {offset_y}")
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracer = ShapeTracer()
    tracer.start()
