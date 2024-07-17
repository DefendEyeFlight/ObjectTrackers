import cv2
from ultralytics import YOLO

model = YOLO("yolov8n_02_07.pt")

video_path = "video.mp4" #set your video path
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or other codecs like 'MP4V'
out = cv2.VideoWriter('Video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, tracker = "bytetrack.yaml", imgsz = [1080,1920]) #you can choose "botsort.yaml" or "bytetrack.yaml"

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()