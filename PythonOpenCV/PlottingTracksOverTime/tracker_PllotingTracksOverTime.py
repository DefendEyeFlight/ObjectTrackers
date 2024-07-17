from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n_02_07.pt")

video_path = "/video_path" #set your video path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video details (fps, width, height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out_path = "video_track.mp4" #set your video output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

track_history = defaultdict(list)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, tracker = "bytetrack.yaml", imgsz = [1080,1920]) #you can choose "botsort.yaml" or "bytetrack.yaml"

        if results is not None:
            for result in results:
                boxes = result.boxes.xywh.cpu()
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()

                    annotated_frame = result.plot()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                        #tracking lines
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=5)

                    out.write(annotated_frame)

        else:
            continue

    else:
        break

cap.release()
out.release()

print(f"Output video saved to: {out_path}")
