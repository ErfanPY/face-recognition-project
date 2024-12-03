import cv2
import face_recognition
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os
from datetime import datetime
import uuid

from face_utils import (
    draw_processed_frame,
    find_faces,
    load_know_images,
    pre_process_frame,
)


def save_face_image(frame, face_location):
    """Save detected face with random ID"""
    # Create images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    # Extract face from frame
    top, right, bottom, left = face_location
    # Scale back up face locations
    scale = 4
    top *= scale
    right *= scale
    bottom *= scale
    left *= scale

    face_image = frame[top:bottom, left:right]

    # Generate filename with random ID
    random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    filename = f"images/unknown_{random_id}.jpg"

    # Save the image
    cv2.imwrite(filename, face_image)
    return filename


def process_faces(frame_queue, result_queue, known_face_encodings, known_face_names):
    """Process function that runs in a separate process"""
    while True:
        try:
            # Get frame data from queue
            frame_data = frame_queue.get()
            if frame_data is None:  # Poison pill for clean shutdown
                break

            # Process faces
            face_locations = np.array(frame_data["face_locations"])

            face_encodings = face_recognition.face_encodings(
                frame_data["rgb_small_frame"], frame_data["face_locations"]
            )

            local_face_names = []
            face_confidences = []
            unknown_encodings = []

            # Process each detected face
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.6
                )
                name = "Unknown"
                confidence = 0.0

                if True in matches:
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        confidence = 1 - face_distances[best_match_index]
                        if confidence > 0.5:
                            name = known_face_names[best_match_index]

                if name == "Unknown":
                    unknown_encodings.append(face_encoding)

                local_face_names.append(name)
                face_confidences.append(confidence)

            print(
                f"Found {len(local_face_names)} faces: {list(zip(local_face_names, face_confidences))}"
            )

            result_queue.put(
                {
                    "names": local_face_names,
                    "confidences": face_confidences,
                    "unknown_encodings": unknown_encodings,
                }
            )

        except Exception as e:
            print(f"Error in process_faces: {e}")
            continue


def main():
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load known faces
    known_face_encodings, known_face_names = load_know_images()

    frame_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue()

    face_process = mp.Process(
        target=process_faces,
        args=(frame_queue, result_queue, known_face_encodings, known_face_names),
    )
    face_process.daemon = True
    face_process.start()

    face_locations = []
    face_names = []
    face_confidences = []
    process_this_frame = True
    paused = False
    current_frame = None

    try:
        while True:
            if not paused:
                ret, frame = video_capture.read()
                if not ret:
                    continue
                current_frame = frame.copy()
            else:
                frame = current_frame.copy()

            if process_this_frame and not paused:
                rgb_small_frame = pre_process_frame(frame)

                face_locations = face_recognition.face_locations(
                    rgb_small_frame, model="hog"
                )

                if face_locations:
                    try:
                        frame_queue.put_nowait(
                            {
                                "face_locations": face_locations,
                                "rgb_small_frame": rgb_small_frame,
                            }
                        )
                        face_names = ["Detecting..."] * len(face_locations)
                        face_confidences = [0.0] * len(face_locations)
                    except mp.queues.Full:
                        pass
                else:
                    face_names = []
                    face_confidences = []

            process_this_frame = not process_this_frame

            # Check for results
            try:
                while True:
                    result = result_queue.get_nowait()
                    face_names = result["names"]
                    face_confidences = result["confidences"]

                    # If we found an unknown face, pause the video
                    if "Unknown" in face_names and not paused:
                        paused = True
                        print(
                            "\nUnknown face detected! Press 's' to save or Enter to skip"
                        )

            except mp.queues.Empty:
                pass

            # Draw the results
            frame_copy = frame.copy()
            for (top, right, bottom, left), name, confidence in zip(
                face_locations, face_names, face_confidences
            ):
                scale = 4
                top *= scale
                right *= scale
                bottom *= scale
                left *= scale

                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 2)
                cv2.rectangle(
                    frame_copy, (left, bottom - 35), (right, bottom), color, cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                conf_text = f"{confidence:.2f}" if confidence > 0 else ""
                cv2.putText(
                    frame_copy,
                    f"{name} {conf_text}",
                    (left + 6, bottom - 6),
                    font,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            # Display the resulting frame
            cv2.imshow("Video", frame_copy)

            key = cv2.waitKey(1) & 0xFF

            # Handle user input when paused
            if paused:
                if key == 13:  # Enter key
                    paused = False
                    print("Skipped")
                elif key == ord("s"):  # 's' key to save
                    # Find the index of the unknown face
                    unknown_idx = face_names.index("Unknown")
                    if unknown_idx >= 0:
                        # Save the face image
                        filename = save_face_image(frame, face_locations[unknown_idx])
                        print(f"Saved face as {filename}")
                        # Reload known faces
                        known_face_encodings, known_face_names = load_know_images()
                        # Update the face process with new encodings
                        frame_queue.put(None)  # Stop the old process
                        face_process.join(timeout=1)
                        # Start new process with updated encodings
                        face_process = mp.Process(
                            target=process_faces,
                            args=(
                                frame_queue,
                                result_queue,
                                known_face_encodings,
                                known_face_names,
                            ),
                        )
                        face_process.daemon = True
                        face_process.start()
                    paused = False

            if key == ord("q"):
                break

    finally:
        # Cleanup
        frame_queue.put(None)
        face_process.join(timeout=1)
        if face_process.is_alive():
            face_process.terminate()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.freeze_support()
    main()
