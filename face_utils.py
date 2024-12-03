import os
import cv2
import face_recognition


def load_know_images(images_dir="images"):
    known_face_encodings = []
    known_face_names = []

    for image in os.listdir(images_dir):
        print(f"{images_dir}/{image}")
        image_file = face_recognition.load_image_file(f"{images_dir}/{image}")
        image_encoding = face_recognition.face_encodings(image_file)[0]

        known_face_encodings.append(image_encoding)
        known_face_names.append(image.split(".")[0])

    return known_face_encodings, known_face_names


def pre_process_frame(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    return rgb_small_frame


def draw_processed_frame(frame, face_locations, face_names, show_gui=True):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    if show_gui:
        # Display the resulting image
        cv2.imshow("Video", frame)
        cv2.waitKey(1)
    return frame

def find_faces(known_face_encodings, known_face_names, rgb_small_frame):
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # # Or instead, use the known face with the smallest distance to the new face
        # face_distances = face_recognition.face_distance(
        #     known_face_encodings, face_encoding
        # )
        # best_match_index = np.argmin(face_distances)
        # if matches[best_match_index]:
        #     name = known_face_names[best_match_index]

        face_names.append(name)
    return face_locations, face_names
