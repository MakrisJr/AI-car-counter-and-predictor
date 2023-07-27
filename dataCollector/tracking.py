import os
from IPython import display
from ultralytics import YOLO
import sys
import yolox
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
import supervision
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
import numpy as np
import cv2
from tqdm import tqdm


MODEL = "yolov8x.pt"

# classes accounted for
CLASS_ID = [2,3,5,7]

HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"{HOME}/data/vehicle-counting.mp4"
TARGET_VIDEO_PATH = f"{HOME}/data/vehicle-counting-result.mp4"

# Horizontal Counter Line
LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)
#test
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def prepare():
    print("Home directory: ", HOME)

    display.clear_output()
    #ultralytics.checks()
    sys.path.append(f"{HOME}/ByteTrack")

    print("yolox.__version__:", yolox.__version__)
    print("supervision.__version__:", supervision.__version__)

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def main():
    model = YOLO(MODEL)
    model.fuse()

    CLASS_NAMES_DICT = model.model.names

    videoInfo = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(videoInfo)

    # add tracker:
    byteTracker = BYTETracker(BYTETrackerArgs())
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    lineCounter = LineCounter(start = LINE_START, end = LINE_END)
    # create annotators
    boxAnnotator = BoxAnnotator(color=ColorPalette(), thickness = 4, text_thickness=4, text_scale=2)
    lineAnnotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    with VideoSink(TARGET_VIDEO_PATH, videoInfo) as sink:
        for frame in tqdm(generator, total=videoInfo.total_frames):
            results = model(frame)
            detections = Detections(
                xyxy = results[0].boxes.xyxy.cpu().numpy(),
                confidence = results[0].boxes.conf.cpu().numpy(),
                class_id = results[0].boxes.cls.cpu().numpy().astype(int)
            )

            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byteTracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            # filtering out detections without trackers
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # updating line counter
            lineCounter.update(detections=detections)
            # annotate and display frame
            frame = boxAnnotator.annotate(frame=frame, detections=detections, labels=labels)
            lineAnnotator.annotate(frame=frame, line_counter=lineCounter)
            sink.write_frame(frame)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

if __name__ == "__main__":
    prepare()
    main()