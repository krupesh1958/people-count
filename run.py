#!/usr/bin/env python3
"""
This script has been created by counting in-out people
Where we use this script:
    Canteen, Hospital etc...
"""
from extension import *


__author__ = "Krupesh1958"    # Author
__version__ = 1.0    # Version


@dataclass
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


# Converts List[STrack] into format that can be consumed
# by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


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

    itr = enumerate(track2detection)
    for tracker_index, detection_index in itr:
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


VIDEO_PATH = "./utils/data/output.mp4"
# Open target video file
with VideoSink(VIDEO_PATH, video_info) as sink:

    itr = tqdm(generator, total=video_info.total_frames)
    for frame in itr:
        # Model prediction on single frame and conversion to supervision Detecitons
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # Filtering out detections with unwanted classes
        mask = np.array(
            [class_id in CLASS_ID for class_id in detections.class_id],
            dtype=bool
        )
        detections.filter(mask=mask, inplace=True)

        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(
            detections=detections, tracks=tracks
        )
        detections.tracker_id = np.array(tracker_id)

        # Filtering out detections without trackers
        mask = np.array(
            [tracker_id is not None for tracker_id in detections.tracker_id], 
            dtype=bool
        )
        detections.filter(mask=mask, inplace=True)

        # Format custom labels
        labels = [
            f"#{xyxy}"
            for xyxy, confidence, class_id, tracker_id
            in detections
        ]

        # Updating line counter
        line_counter.update(detections=detections)

        # Annotation and display frame
        frame = box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        sink.write_frame(frame)
