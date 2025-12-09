# Import the Picamera2 module to access the Raspberry Pi Camera
from picamera2 import Picamera2

# Import OpenCV for image processing and drawing graphics on frames
import cv2

# Import the FER emotion detector class
# Note: on Raspberry Pi, FER must be imported from fer.fer (not fer)
from fer.fer import FER

# Import time to allow a short delay while initializing the camera
import time

def main():
    # ---------------------------
    # CAMERA INITIALIZATION
    # ---------------------------

    # Create a Picamera2 object to communicate with the Pi Camera Module 3
    picam2 = Picamera2()

    # Configure the camera's resolution.
    # 640x480 is chosen because it is fast enough for real-time emotion detection.
    config = picam2.create_preview_configuration(main={"size": (640, 480)})

    # Apply configuration settings to the camera
    picam2.configure(config)

    # Start the camera stream so frames can be captured in real time
    picam2.start()

    # Small delay (1 second) to give the camera time to adjust exposure
    time.sleep(1)


    # ---------------------------
    # EMOTION DETECTOR SETUP
    # ---------------------------

    # Create an FER emotion detector object
    # mtcnn=False uses OpenCV face detection, which is faster on Raspberry Pi
    detector = FER(mtcnn=False)

    print("Emotion detection started — press 'q' to quit")


    # ---------------------------
    # MAIN PROCESSING LOOP
    # ---------------------------

    while True:
        # Capture a single frame from the Pi camera as a NumPy array
        frame = picam2.capture_array()

        # FER expects images in RGB format, but OpenCV uses BGR
        # Convert BGR → RGB before running emotion detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run FER's emotion detector on the RGB frame
        # This returns a list of detected faces, bounding boxes, and emotion scores
        results = detector.detect_emotions(rgb)

        # Loop through each detected face and draw results on the frame
        for r in results:
            # Extract the bounding box (x, y = top-left corner, w = width, h = height)
            (x, y, w, h) = r["box"]

            # Extract the dictionary of emotion probabilities
            emotions = r["emotions"]

            # Determine which emotion has the highest confidence value
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Write the emotion label + confidence above the bounding box
            cv2.putText(frame,
                        f"{top_emotion}: {confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)


        # Display the processed frame in a window titled "Emotion Detection"
        cv2.imshow("Emotion Detection (press q to quit)", frame)

        # Listen for a key press.
        # If the user presses 'q', the demo will stop and the window closes.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # ---------------------------
    # CLEANUP SECTION
    # ---------------------------

    # Stop the camera from streaming frames
    picam2.stop()

    # Close all OpenCV windows that were opened
    cv2.destroyAllWindows()

    print("Emotion detection stopped.")


# This statement ensures the main function only runs
# when the script is executed directly (not imported)
if __name__ == "__main__":
    main()
