#!/usr/bin/env python3
"""Initialize supervision required librarys"""
from __future__ import annotations

from extensions import *


# Source video frame
cv.imwrite("./src/utils/data/test_1_frame.jpg", frame)

# Initialize polygon zone
polygon = np.array([
    [0, 406],[1200, 398]
])

video_info = sv.VideoInfo.from_video_path("./src/utils/data/test_1.mp4")
zone = sv.PolygonZone(
    polygon=polygon,
    frame_resolution_wh=video_info.resolution_wh
)

box_annotator = sv.BoxAnnotator(
    thickness=4,
)

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.black(),
    thickness=6,
)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=800)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[detections.class_id == 0]
    zone.trigger(detections=detections)

    box_annotator = sv.BoxAnnotator(
        thickness=1
    )

    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
    )
    frame = zone_annotator.annotate(scene=frame)

    return frame

sv.process_video(
    source_path=VIDEO_PATH,
    target_path=f"./src/utils/data/test_output.mp4",
    callback=process_frame
)
