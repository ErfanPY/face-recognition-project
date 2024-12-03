import cv2
import face_recognition
import numpy as np
import requests
import multiprocessing as mp
from multiprocessing import shared_memory
# from flask import Flask, Response
import socket

from face_utils import (
    draw_processed_frame,
    find_faces,
    load_know_images,
    pre_process_frame,
)
# # Initialize Flask app for streaming the processed frames
# app = Flask(__name__)

# # Flask route to serve the processed frames as an MJPEG stream
# @app.route('/stream_processed')
# def stream_processed():
#     def generate():
#         global processed_frame
#         while True:
#             if processed_frame is not None:
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Function to run Flask server in a separate thread
# def start_flask_app():
#     app.run(host='0.0.0.0', port=5001)


def process_faces(frame_queue, result_queue, known_face_encodings, known_face_names):
    """Process function that runs in a separate process"""
    while True:
        try:
            # Get frame data from queue
            frame_data = frame_queue.get()
            if frame_data is None:  # Poison pill for clean shutdown
                break

            # Process faces
            face_encodings = face_recognition.face_encodings(
                frame_data["rgb_small_frame"], frame_data["face_locations"]
            )

            local_face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                local_face_names.append(name)
            print(f"Found faces: {local_face_names}")
            # Put results in queue
            result_queue.put(local_face_names)

        except Exception as e:
            print(f"Error in process_faces: {e}")
            continue

def main():
    # Replace with the IP address of your Raspberry Pi
    raspberry_pi_ip = "192.168.76.120"  # Replace with your Raspberry Pi's actual IP address
    video_stream_url = f"http://{raspberry_pi_ip}:5000/video_feed"

    # Initialize video capture
    video_capture = cv2.VideoCapture(video_stream_url)

    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    # Load known faces
    known_face_encodings, known_face_names = load_know_images()

    # Create queues for inter-process communication
    frame_queue = mp.Queue(
        maxsize=1
    )  # Limit queue size to ensure we process recent frames
    result_queue = mp.Queue()

    # Start the face processing process
    face_process = mp.Process(
        target=process_faces,
        args=(frame_queue, result_queue, known_face_encodings, known_face_names),
    )
    face_process.daemon = True
    face_process.start()

    face_locations = []
    face_names = []
    last_frame_had_faces = False

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            # Process frame
            rgb_small_frame = pre_process_frame(frame)
            face_locations = face_recognition.face_locations(rgb_small_frame)

            # If we found faces in this frame
            if face_locations:
                # If we didn't have faces in the last frame or we're not currently processing
                if not last_frame_had_faces:
                    try:
                        # Try to add new frame to queue without blocking
                        frame_queue.put_nowait(
                            {
                                "face_locations": face_locations,
                                "rgb_small_frame": rgb_small_frame,
                            }
                        )
                        face_names = ["Checking..."] * len(face_locations)
                    except mp.queues.Full:
                        # Queue is full, skip this frame
                        pass
                last_frame_had_faces = True
            else:
                face_names = []
                last_frame_had_faces = False

            # Check for results
            try:
                while True:  # Drain the queue of any old results
                    new_face_names = result_queue.get_nowait()
                    face_names = new_face_names  # Keep only the most recent result
            except mp.queues.Empty:
                pass

            # Draw the results
            draw_processed_frame(
                frame=frame,
                face_locations=face_locations,
                face_names=face_names,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        frame_queue.put(None)  # Send poison pill
        face_process.join(timeout=1)
        if face_process.is_alive():
            face_process.terminate()
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # This is required for Windows support
    mp.freeze_support()
    main()
