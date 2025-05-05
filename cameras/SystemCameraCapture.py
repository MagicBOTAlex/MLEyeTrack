import struct
import cv2
import numpy as np
import queue
import threading
import time
from colorama import Fore
from enum import Enum
import socket

from cameras.ICameraSource import ICameraSource

# Should not be subject to Babble's license.
# This is derived from my code at: https://github.com/MagicBOTAlex/EyeTrackVR/blob/v2.0-beta-feature-branch/EyeTrackApp/Camera/SystemCamera.py
# This is before they changed to their new license

class SystemCamera(ICameraSource):
    def __init__(self, source=0):
        """
        :param source: OpenCV capture source (device index or URL)
        """
        super().__init__()
        self.source = source
        self.cv2_capture = None

    def _update(self):
        # Try to open the capture
        self.cv2_capture = cv2.VideoCapture(int(self.source))
        
        timeout   = 5.0    # total time to wait, in seconds
        interval  = 0.1    # how often to check, in seconds
        start     = time.time()
        while time.time() - start < timeout:
            if self.cv2_capture.isOpened():
                print("Camera opened successfully!")
                break
            time.sleep(interval)
        else:
            print(f"Timed out after {timeout} seconds waiting for condition.")

        # Read loop
        while self.running:
            ret, frame = self.cv2_capture.read()
            if not ret:
                # failed to grab a frame
                break

            # Store the latest frame (thread-safe)
            with self.lock:
                self.frame = frame

        # Clean up
        self.cv2_capture.release()