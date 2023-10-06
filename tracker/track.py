"""Track every project"""
from __future__ import annotations


class TrackableObject:
	
    def __init__(self,  object_id: int, centroid: object) -> None:
        self.object_id: int = object_id
        self.centroids: object = [centroid]
        self.counted: bool = False
