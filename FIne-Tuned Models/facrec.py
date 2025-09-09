from asyncio.windows_events import NULL

import cv2
import face_recognition
import numpy as np
import time
import logging
import requests
from pathlib import Path

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
ESP32_CAM_URL = "http://192.168.118.217/capture"  # ESP32-CAM snapshot URL
BASE_FOLDER = Path(r"C:\Users\shshv\PycharmProjects\voice_test\known_faces")
CONFIDENCE_THRESHOLD = 0.55
FRAME_SKIP = 2  # Process every 2nd frame for efficiency
SCAN_DURATION = 1  # Print detection results every 3 seconds
faces = NULL
class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        self.frame_count = 0
        self.last_detection_time = time.time()

    def load_known_faces(self):
        """Load known faces from the specified folder."""
        logger.info("Loading known faces...")
        try:
            valid_extensions = {".jpg", ".jpeg", ".png"}
            for person_folder in BASE_FOLDER.iterdir():
                if person_folder.is_dir():
                    person_name = person_folder.name
                    logger.info(f"Processing folder: {person_name}")

                    for image_path in person_folder.glob("*"):
                        if image_path.suffix.lower() in valid_extensions:
                            try:
                                image = face_recognition.load_image_file(str(image_path))
                                face_locations = face_recognition.face_locations(image, model="hog")

                                if face_locations:
                                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                                    self.known_face_encodings.append(face_encoding)
                                    self.known_face_names.append(person_name)
                                    logger.info(f"Loaded {image_path.name}")
                                else:
                                    logger.warning(f"No face found in {image_path.name}")

                            except Exception as e:
                                logger.error(f"Error processing {image_path.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")
            exit()

    def get_frame_from_esp32(self):
        """Fetch a frame from the ESP32-CAM."""
        try:
            response = requests.get(ESP32_CAM_URL, timeout=2)  # Fetch frame from ESP32
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return True, frame
            else:
                logger.error("Failed to fetch frame from ESP32-CAM")
                return False, None
        except requests.exceptions.RequestException as e:
            logger.error(f"ESP32-CAM Connection Error: {e}")
            return False, None

    def run(self):
        """Run the face recognition system."""
        logger.info(f"Starting face recognition with {len(self.known_face_encodings)} known faces.")

        while True:
            ret, frame = self.get_frame_from_esp32()
            if not ret:
                logger.error("Failed to capture frame from ESP32-CAM.")
                continue

            self.frame_count += 1

            # Skip frames for efficiency
            if self.frame_count % FRAME_SKIP != 0:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Detect faces
            face_locations = face_recognition.face_locations(frame, model="hog")

            if face_locations and time.time() - self.last_detection_time >= SCAN_DURATION:
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                self.process_faces(frame, face_locations, face_encodings)
                self.last_detection_time = time.time()

            # Ensure bounding boxes & text are visible
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def process_faces(self, frame, face_locations, face_encodings):
        """Recognize faces, print detected names, and draw bounding boxes."""
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=CONFIDENCE_THRESHOLD)
            name = "Unknown"
            confidence = 0

            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

            # Print detected name every 3 seconds
            logger.info(f"Detected: {name} (Confidence: {confidence:.2%})")

            # Draw a rectangle around the face
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Display the name and confidence above the rectangle
            cv2.putText(frame, f"{name} ({confidence:.2%})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


if __name__ == "__main__":
    try:
        face_system = FaceRecognitionSystem()
        face_system.run()
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
