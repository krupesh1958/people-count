"""Initialize centroid tracker"""
from __future__ import annotations

import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from typing import DefaultDict


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50) -> None:
        """Initlaize the centroid instace variables"""
        self.obj_id: int = 0
        self.objects: DefaultDict = OrderedDict()
        self.disappeared: DefaultDict = OrderedDict()
        self.max_disappeared: int = max_disappeared
        self.max_distance: int = max_distance


    def register(self, centroid):
        """
        When registering an object we use the next available object
        ID to store the centroid
        """
        self.objects[self.obj_id] = centroid
        self.disappeared[self.obj_id] = 0
        self.obj_id += 1
    

    def deregister(self, id):
        """
        To deregister an object ID we delete the object ID from
        both of our respective dictionaries
        """
        del self.objects[id]
        del self.disappeared[id]
    

    def update(self, rects):
        """
        Check to see if the list of input bounding box rectangles
		is empty
        """
        if len(rects) == 0:
            for itr in list(self.disappeared.keys()):
                self.disappeared[itr] += 1
                if self.disappeared[itr] > self.max_disappeared:
                    self.deregister(itr)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            centroid_x = int((start_x + end_x) / 2.0)
            centroid_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (centroid_x, centroid_y)

        if len(self.objects) == 0:
          for i in range(0, len(input_centroids)):
            self.register(input_centroids[i])

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            centroid_d = dist.cdist(np.array(object_centroids), input_centroids)
            rows = centroid_d.min(axis=1).argsort()
            cols = centroid_d.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                   continue
                if centroid_d[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_cols.add(col), used_rows.add(row)
            
            unused_r = set(range(0, centroid_d.shape[0])).difference(used_rows)
            unused_c = set(range(0, centroid_d.shape[1])).difference(used_cols)

            if centroid_d.shape[0] >= centroid_d.shape[1]:
                for row in unused_r:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1

                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
            
            else:
                for col in unused_c:
                    self.register(input_centroids[col])
                
        return self.objects

