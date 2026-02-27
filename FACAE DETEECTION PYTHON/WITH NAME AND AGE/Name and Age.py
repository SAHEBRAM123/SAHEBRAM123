import face_recognition
import cv2

# Load known faces
known_image = face_recognition.load_image_file("person1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load test image
frame = cv2.imread("test.jpg")
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces([known_encoding], face_encoding)
    name = "SAHEBRAM MANDI"
    if True in matches:
        name = "Person 1"

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Result", frame)
cv2.waitKey(0)
