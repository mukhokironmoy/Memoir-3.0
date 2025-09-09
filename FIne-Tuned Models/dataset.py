import cv2
import os
import time
from datetime import datetime


def capture_image_burst(save_folder, num_images=10, delay=0.5):
    """
    Captures a burst of images from the camera and saves them to the specified folder.

    Args:
        save_folder (str): Path to save the captured images
        num_images (int): Number of images to capture in the burst
        delay (float): Delay between captures in seconds
    """
    # Create the full path for saving images
    save_path = os.path.join(save_folder, 'known_faces', 'keith')

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Starting burst capture of {num_images} images...")

    # Capture frames
    for i in range(num_images):
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Failed to capture image {i + 1}")
            continue

        # Generate filename with timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"shashvat_{timestamp}.jpg"
        full_path = os.path.join(save_path, filename)

        # Save the image
        cv2.imwrite(full_path, frame)
        print(f"Captured image {i + 1}/{num_images}: {filename}")

        # Display the image (press 'q' to quit early)
        cv2.imshow('Burst Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait for the specified delay
        time.sleep(delay)

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Burst capture complete. {num_images} images saved to {save_path}")


if __name__ == "__main__":
    # Replace this with your base folder path
    base_folder = "."  # Current directory, change as needed

    # Call the function
    capture_image_burst(base_folder, num_images=10, delay=0.5)