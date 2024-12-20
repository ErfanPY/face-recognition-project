import cv2
import face_recognition
import numpy as np

# Replace with the IP address of your Raspberry Pi
raspberry_pi_ip = "192.168.76.120"  # Replace with your Raspberry Pi's actual IP address
video_stream_url = f"http://{raspberry_pi_ip}:5000/video_feed"

# Open the video stream from Raspberry Pi
video_capture = cv2.VideoCapture(video_stream_url)

# Load sample pictures and learn how to recognize them
obama_image = face_recognition.load_image_file("images/Erfan.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("images/Alireza.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
]
known_face_names = [
    "Erfan",
    "Alireza",
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
DO_SHOW_GUI = True

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture frame. Check if Raspberry Pi stream is accessible.")
        break

    if process_this_frame:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    if DO_SHOW_GUI:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print(face_names)
video_capture.release()
cv2.destroyAllWindows()
