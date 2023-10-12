#!/usr/bin/env python3
"""Counting people in-out process"""
from __future__ import annotations

import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack

from typing import List
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from ultralytics import YOLO
from tqdm import tqdm
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = np.array([track.tlbr for track in tracks], dtype=float)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


# Create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
model = YOLO("/Users/yudiz/Desktop/face_detection/people-count/apps/utils/models/yolov8s.pt")
generator = get_video_frames_generator("/Users/yudiz/Desktop/face_detection/people-count/apps/utils/data/test_iphone.MOV")
iterator = iter(generator)
frame = next(iterator)
results = model(frame)

# For counting humans
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [0]
detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)
# Format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections
]
# Annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)


# Setting
LINE_START = Point(17, 1250)
LINE_END = Point(1620, 1250)

VideoInfo.from_video_path("/Users/yudiz/Desktop/face_detection/people-count/apps/utils/data/test_iphone.MOV")


# Create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# Create VideoInfo instance
video_info = VideoInfo.from_video_path("/Users/yudiz/Desktop/face_detection/people-count/apps/utils/data/output.MOV")
line_counter = LineCounter(start=LINE_START, end=LINE_END)
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=1)

VIDEO_PATH = "/Users/yudiz/Desktop/face_detection/people-count/apps/utils/data/iphone_target.mp4"
with VideoSink(VIDEO_PATH, video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # updating line counter
        line_counter.update(detections=detections)
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        sink.write_frame(frame)
