
import time

import sys
sys.path.append('/home/ichnaea/ichnaea/src')  # relative import 2 folders up didn't work

from hardware.nano.stepper_driver import move_stepper
from motion_estimation.pid import PID
###############################################################################################

from multiprocessing import Value, Process, Manager

import signal

import os
import pathlib
import imutils
import time
import cv2
from imutils.video import FPS
import traceback

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    # "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    #"tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create,
}

class ObjectTracker:

    def __init__(self, model_name="kcf"):
        self.tracker = None
        self.model_name = model_name

    def start_tracker(self, bounding_box, image):
        print("Starting tracker for bounding box:", bounding_box)
        self.tracker = OPENCV_OBJECT_TRACKERS[self.model_name]()
        self.tracker.init(image, bounding_box)

    def track(self, bounding_box, image):
        (H, W) = image.shape[:2] 
        if not bounding_box:
            raise("No bounding box found, rerun object detection...")
        if not self.tracker:
            raise("Tracker not initialized...")

        (success, box) = self.tracker.update(image)

        return image,success,box

