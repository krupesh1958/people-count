"""Helper function to get input from user side with video detection"""
from __future__ import annotations

import argparse


def parse_arguments() -> argparse[object]:
    """Function to parse the arguments"""
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", type=str, required=True,
                    help="Path to optional input video file")
    
    ap.add_argument("-m", "--model", required=True,
                    help="Path to yolo model")

    args = vars(ap.parse_args())
    return args
