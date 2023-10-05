# -*- coding: utf-8 -*-
"""
Extensions module. Each extension is initialized in the app
factory located in run.py
"""
from __future__ import annotations

import json
import datetime
import logging
import argparse
import csv
import cv2 as cv
import numpy as np
import time
import dlib

from itertools import zip_longest
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import resize


# Config logger
logging.basicConfig(
    level=logging.INFO,
    format="[INFO]%(message)s"
)
logger = logging.getLogger(__name__)


# Start frame
frames = FPS().start()


writer = None
total_frames = total_down = total_up = 0
trackable_objects = dict()
tracker, total, move_in, move_out, in_time, out_time = ([],) * 6


__all__ = [
    "time",
    "logger",
    "dlib",
    "VideoStream",
    "config",
    "argparse",
    "zip_longest",
    "csv",
    "cv",
    "np",
    "resize",
    "total_frames",
    "trackable_objects",
    "tracker",
    "total",
    "move_in",
    "move_out",
    "in_time",
    "out_time",
    "datetime",
    "total_frames",
    "total_down",
    "total_up",
    "time",
    "writer",
]
