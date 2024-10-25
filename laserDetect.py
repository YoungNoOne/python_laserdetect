import cv2
import numpy as np

# Define the HSV color range for detecting a red laser pointer.
# You may need to adjust these values based on your laser's color and brightness.
LOWER_RED = np.array([160, 100, 100])
UPPER_RED = np.array([180, 255, 255])

# Constants for distance calculation
KNOWN_DISTANCE = 2.0  # meters (adjust based on your setup)
FOCAL_LENGTH = 700    # example focal length, needs to be calibrated

def calculate_distance(detected_radius):
    """
    Estimate distance to the laser pointer based on the detected radius.
    The focal length needs to be calibrated for this to be accurate.
    """
    if detected_radius > 0:
        return FOCAL_LENGTH / detected_radius
    else:
        return None

def process_frame(frame):
    """
    Processes the input video frame to detect a red laser point and estimate its distance.
    """
    # Convert the frame from BGR to HSV color space.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to filter out everything except the red color.
    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)

    # Apply Gaussian Blur to reduce noise and improve detection accuracy.
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour, which likely corresponds to the laser point.
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the minimum enclosing circle of the largest contour.
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Filter out small contours to avoid detecting noise.
        if radius > 5:
            # Draw a circle around the detected laser point.
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

            # Estimate the distance to the laser point.
            distance = calculate_distance(radius)
            
            # Check if the laser pointer is within the specified range (1.9 to 2.1 meters).
            if distance and 1.9 <= distance <= 2.1:
                # Display the distance on the video frame.
                cv2.putText(frame, f"Laser Detected at {distance:.2f} meters", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def main():
    # Start video capture from the default camera (index 0).
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Continuously capture frames from the webcam.
    while True:
        # Read a frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame to detect the laser pointer and display the distance.
        processed_frame = process_frame(frame)

        # Show the processed frame in a window.
        cv2.imshow("Laser Pointer Detection", processed_frame)

        # Break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
