def analyze_screenshot(self, screenshot):
    # Define regions for current and next button
    current_button_region = screenshot[750:850, 800:900]  # Adjust coordinates for blue-circled button
    next_button_region = screenshot[750:850, 900:1000]    # Adjust coordinates for next button
    
    # Analyze both regions
    current_button = self.check_button_state(current_button_region)
    next_button = self.check_button_state(next_button_region)
    
    return {
        'current_button': current_button,
        'next_button': next_button
    }