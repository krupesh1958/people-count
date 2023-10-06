"""Common functions"""
from __future__ import annotations

from extensions import argparse, zip_longest, csv


def parse_arguments() -> argparse[object]:
    """Function to parse the arguments"""
    ap = argparse.ArgumentParser()

    ap.add_argument("-p", "--prototxt", required=True,
                    help="Path to Caffe `deploy` protxt file")

    ap.add_argument("-m", "--model", required=True,
                    help="Path to Caffe pre-trained model")
    
    ap.add_argument("-i", "--input", type=str,
                    help="Path to optional input video file")
    
    ap.add_argument("-o", "--output", type=str,
                    help="Path to optional output video file")

    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="Minimum probability to filter weak detections")

    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
	
    ap.add_argument("-u", "--url", required=False,
                    help="Live video tracking shoud to in url")
    
    args = vars(ap.parse_args())
    return args

