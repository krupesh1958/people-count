"""Initialize required extensions"""
import cv2 as cv
import numpy as np
import supervision as sv

from ultralytics import YOLO


VIDEO_PATH = "./src/utils/data/test_1.mp4"

model = YOLO("./src/utils/models/yolov8s.pt")

generator = sv.get_video_frames_generator("./src/utils/data/test_1.mp4")
iterator = iter(generator)
frame = next(iterator)


__all__ = [
    "cv",
    "np",
    "sv",
    "YOLO",
    "VIDEO_PATH",
    "frame",
    "model"
]
