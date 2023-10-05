"""Find human body with centroid use"""
from __future__ import annotations

import numpy as np

from collections import OrderedDict
from scipy.spatial import distance as dist
from dataclasses import dataclass
from typing import (
    Any,
    DefaultDict,
    Optional
)


@dataclass
class CentroidTracker:
    """
    Initialize the next unique object ID along with 
    two ordered dictionaries used to keep track of 
    mapping a given object ID to its centroid and number
    of consecutive frames it has been marked as "disappeared",
    respectively
    """
    next_object_id = 0
    objects = OrderedDict()
    disappeared = OrderedDict()
    max_disappeared = 50    # Store the number of maximum consecutive
    max_distance = 50    # Store the maximum distance between centroids


    def register(self, centroid: object) -> None:
        """
        Registering an object we use the next available object
        ID to store the centroid
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return
