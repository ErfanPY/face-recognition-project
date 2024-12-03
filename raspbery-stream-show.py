import cv2

# Replace with the IP address of your Raspberry Pi
raspberry_pi_ip = (
    "localhost"  # Replace with your Raspberry Pi's actual IP address
)
video_stream_url = f"http://{raspberry_pi_ip}:5000/display_processed"

# Open the video stream from Raspberry Pi
video_capture = cv2.VideoCapture(video_stream_url)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture frame. Check if Raspberry Pi stream is accessible.")
        break

    # # Resize frame for faster processing
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb_small_frame = small_frame[:, :, ::-1]


    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
