import cv2
import numpy as np

def detect_water_level(frame):
    """
    Detect water level in a bottle from a frame
    Returns the fill percentage and processed frame
    """
    # Resize frame for consistent processing
    resized = cv2.resize(frame, (400, 600))
    
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, resized, None
    
    # Find the largest contour (assumed to be the bottle)
    bottle_contour = max(contours, key=cv2.contourArea)
    
    # Filter out very small contours
    if cv2.contourArea(bottle_contour) < 1000:
        return None, resized, None
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(bottle_contour)
    
    # Extract region of interest (bottle area)
    roi = resized[y:y+h, x:x+w]
    
    if roi.size == 0:
        return None, resized, (x, y, w, h)
    
    # Convert ROI to grayscale and threshold
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Detect water level by finding the lowest row with significant white pixels
    heights = []
    for i in range(thresh.shape[0]):
        row = thresh[i, :]
        if cv2.countNonZero(row) > 0.5 * thresh.shape[1]:
            heights.append(i)
    
    if heights:
        water_level = max(heights)
        fill_ratio = (thresh.shape[0] - water_level) / thresh.shape[0]
        percentage = fill_ratio * 100
        return percentage, resized, (x, y, w, h)
    else:
        return 0, resized, (x, y, w, h)

def main():
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Water Bottle Fill Level Detector")
    print("Instructions:")
    print("- Hold the water bottle in front of the camera")
    print("- Make sure the bottle is well-lit and clearly visible")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    
    frame_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process every 3rd frame to improve performance
        if frame_count % 3 == 0:
            # Detect water level
            percentage, processed_frame, bbox = detect_water_level(frame)
            
            # Create display frame
            display_frame = processed_frame.copy()
            
            # Draw bounding box if bottle detected
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw water level indicator
                if percentage is not None:
                    # Calculate water level position
                    water_y = int(y + h * (1 - percentage/100))
                    cv2.line(display_frame, (x, water_y), (x+w, water_y), (255, 0, 0), 3)
                    
                    # Add text with fill percentage
                    text = f"Fill Level: {percentage:.1f}%"
                    cv2.putText(display_frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add status text based on fill level
                    if percentage > 80:
                        status = "FULL"
                        color = (0, 255, 0)  # Green
                    elif percentage > 50:
                        status = "HALF FULL"
                        color = (0, 255, 255)  # Yellow
                    elif percentage > 20:
                        status = "LOW"
                        color = (0, 165, 255)  # Orange
                    else:
                        status = "EMPTY"
                        color = (0, 0, 255)  # Red
                    
                    cv2.putText(display_frame, status, (x, y+h+25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # No bottle detected
                cv2.putText(display_frame, "No bottle detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add instructions
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save", (10, display_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:
            display_frame = cv2.resize(frame, (400, 600))
        
        # Display the frame
        cv2.imshow('Water Bottle Fill Level Detector', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"water_bottle_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        
        frame_count += 1
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    main()