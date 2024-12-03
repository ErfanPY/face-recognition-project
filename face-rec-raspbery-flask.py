import cv2
import face_recognition
import numpy as np
import multiprocessing as mp

from face_utils import (
    draw_processed_frame,
    find_faces,
    load_know_images,
    pre_process_frame,
)



def process_faces(frame_queue, result_queue, known_face_encodings, known_face_names):
    """Process function that runs in a separate process"""
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                break

            face_encodings = face_recognition.face_encodings(
                frame_data["rgb_small_frame"], frame_data["face_locations"]
            )

            local_face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                local_face_names.append(name)
            result_queue.put(local_face_names)

        except Exception as e:
            print(f"Error in process_faces: {e}")
            continue


def main():
    raspberry_pi_ip = "192.168.76.120"
    video_stream_url = f"http://{raspberry_pi_ip}:5000/video_feed"

    video_capture = cv2.VideoCapture(video_stream_url)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    known_face_encodings, known_face_names = load_know_images()

    frame_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue()

    face_process = mp.Process(
        target=process_faces,
        args=(frame_queue, result_queue, known_face_encodings, known_face_names),
    )
    face_process.start()

    face_locations = []
    face_names = []
    last_frame_had_faces = False

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            rgb_small_frame = pre_process_frame(frame)
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if face_locations:
                if not last_frame_had_faces:
                    try:
                        frame_queue.put_nowait({
                            "face_locations": face_locations,
                            "rgb_small_frame": rgb_small_frame,
                        })
                        face_names = ["Checking..."] * len(face_locations)
                    except mp.queues.Full:
                        pass
                last_frame_had_faces = True
            else:
                face_names = []
                last_frame_had_faces = False

            try:
                while True:
                    new_face_names = result_queue.get_nowait()
                    face_names = new_face_names
            except mp.queues.Empty:
                pass

            processed_frame = draw_processed_frame(
                frame=frame,
                face_locations=face_locations,
                face_names=face_names,
                show_gui=True
            )

    finally:
        frame_queue.put(None)
        face_process.join(timeout=1)
        if face_process.is_alive():
            face_process.terminate()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.freeze_support()
    main()
