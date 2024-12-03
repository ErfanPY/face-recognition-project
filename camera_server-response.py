from flask import Flask, Response, request
import cv2

app = Flask(__name__)

# Initialize the USB camera (use 0 for the first connected camera)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    """Generate frames from the USB camera for real-time processing."""
    while True:
        ret, frame = camera.read()  # Read frame from the camera
        if not ret:
            continue
        _, buffer = cv2.imencode(".jpg", frame)  # Encode the frame as JPEG
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    """Route for the live video feed."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# Endpoint to receive and display processed frames
processed_frame = None

@app.route("/processed_frame", methods=["POST"])
def receive_processed_frame():
    """Receive processed frame data for display."""
    global processed_frame
    processed_frame = request.data  # Store received frame data
    return "Frame received", 200

@app.route("/display_processed")
def display_processed():
    """Route for displaying processed frames."""
    def generate_processed():
        global processed_frame
        while True:
            if processed_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + processed_frame + b"\r\n"
                )
            else:
                pass  # Optionally yield a placeholder or wait briefly if no frame is available

    return Response(
        generate_processed(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
