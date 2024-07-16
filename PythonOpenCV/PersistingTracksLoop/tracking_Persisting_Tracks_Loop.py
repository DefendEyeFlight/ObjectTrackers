import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n_02_07.pt")

# Open the video file
video_path = "footballlll.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or other codecs like 'MP4V'
out = cv2.VideoWriter('Video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, output video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()