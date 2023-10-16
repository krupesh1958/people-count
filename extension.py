# -*- coding: utf-8 -*-
"""
Extensions module. Each extension is initialized in the app
factory located in run.py
"""
from __future__ import annotations

import numpy as np
from apps.couting.ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack

from tqdm import tqdm
from typing import List
from ultralytics import YOLO
from dataclasses import dataclass
from supervision.video.sink import VideoSink
from supervision.draw.color import ColorPalette
from onemetric.cv.utils.iou import box_iou_batch
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import (
    LineCounter,
    LineCounterAnnotator
)

from helper import parse_arguments


args = parse_arguments()

@dataclass
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# Create instance of BoxAnnotator
box_annotator = BoxAnnotator(
    color=ColorPalette(),
    thickness=4,
    text_thickness=4,
    text_scale=2
)
model = YOLO(args["model"])
model.fuse()

# Create frame and results with models
generator = get_video_frames_generator(args["input"])
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
frame = box_annotator.annotate(
    frame=frame,
    detections=detections,
    labels=labels
)

# Setting
LINE_START = Point(0, 1200)
LINE_END = Point(1300, 1200)

VideoInfo.from_video_path(args["input"])


# Create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# Create VideoInfo instance
video_info = VideoInfo.from_video_path(args["input"])
line_counter = LineCounter(start=LINE_START, end=LINE_END)
box_annotator = BoxAnnotator(
    color=ColorPalette(),
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)
line_annotator = LineCounterAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2
)

__all__ = [
    "annotations",
    "np",
    "BYTETracker",
    "STrack",
    "tqdm",
    "List",
    "YOLO",
    "dataclass",
    "VideoSink",
    "ColorPalette",
    "box_iou_batch",
    "Point",
    "VideoInfo",
    "get_video_frames_generator",
    "Detections",
    "BoxAnnotator",
    "LineCounter",
    "LineCounterAnnotator",
    "parse_arguments",
    "video_info",
    "generator",
    "model",
    "CLASS_ID",
    "byte_tracker",
    "line_counter",
    "box_annotator",
    "line_annotator",
    "args"
]
