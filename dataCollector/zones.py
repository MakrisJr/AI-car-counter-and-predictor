import numpy as np
import supervision as sv
import cv2
from pathlib import Path
import os
from tracker import ZoneTracker
from ultralytics import YOLO
from supervision.draw.utils import draw_text

model = YOLO('yolov8s.pt')

VIDEO = os.path.join(os.getcwd(), "test_video2.mp4")

video_info = sv.VideoInfo.from_video_path(VIDEO)

# extract video frame
generator = sv.get_video_frames_generator(VIDEO)
iterator = iter(generator)

frame = next(iterator)

# save first frame
cv2.imwrite("first_frame.png", frame)

tracker = ZoneTracker(model, video_info)
results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_yolov8(results)
#only keep detections for 'car', id = 2
detections = detections[detections.class_id == 2] 

tracker.update(frame, detections)
frame = tracker.getAnnotatedFrame()


cv2.imwrite("annotated_frame.png", frame)

def process_frame(frame: np.ndarray, i) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)

    #only keep detections for 'car', id = 2
    detections = detections[detections.class_id == 2] 

    tracker.update(frame, detections)
    frame = tracker.getAnnotatedFrame()
    print("Count: ", tracker.getCount())
    #cv2.imshow("frame: ", frame)
    text = f"count: {tracker.getCount()}"
    frame = draw_text(frame, text, sv.Point(x=50, y=50))
    return frame


sv.process_video(source_path=VIDEO, target_path=f"result.mp4", callback=process_frame)

