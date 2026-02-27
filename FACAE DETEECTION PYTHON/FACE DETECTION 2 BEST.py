import cv2

def main():
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture (0 = default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("✅ Webcam started. Press 'q' to quit.")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame.")
            break

        # Convert frame to grayscale (needed for Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)   # Adjusted for better accuracy
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show output window
        cv2.imshow("Face Detection", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
