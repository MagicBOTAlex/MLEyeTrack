#!/usr/bin/env python3
import os
import json
import time
import logging
import threading
from queue import Queue, Empty
import numpy as np
import cv2
from inference import InferenceTask
from osc import OSCSenderTask
from helpers import *
from config import ConfigTask
from capture import CaptureTask
from cameras.CameraFactory import CameraFactory

# ----------------------------
# Main
# ----------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    shared_cfg = {}
    cfg_lock = threading.Lock()

    # start config watcher
    ConfigTask("./Settings.json", shared_cfg, cfg_lock).start()
    # wait for initial config
    while True:
        with cfg_lock:
            if shared_cfg:
                break
        time.sleep(0.1)
        
    # Configs
    with cfg_lock:
        cfg = dict(shared_cfg)

    # queues
    qL, qR = Queue(maxsize=1), Queue(maxsize=1)
    results = Queue(maxsize=5)

    # Setup models before cameras
    InferenceTask(cfg, qL, qR, results, shared_cfg, cfg_lock).start()
    logging.info("Models loaded and cameras ready.")

    # setup camera
    # capL = MJPEGVideoCapture(f"http://{cfg['leftEye']}"); capL.open()
    # capR = MJPEGVideoCapture(f"http://{cfg['rightEye']}"); capR.open()
    # capL = SystemCamera(0); capL.open()
    # capR = SystemCamera(1); capR.open()
    capL = CameraFactory.get_camera_from_string_type(cfg['leftEye']); capL.open()
    capR = CameraFactory.get_camera_from_string_type(cfg['rightEye']); capR.open()
    logging.info("Waiting for at least one camera to become ready…")
    while not (capL.isPrimed() or capR.isPrimed()):
        time.sleep(0.1)
    logging.info(
        "Camera ready. leftPrimed=%s, rightPrimed=%s",
        capL.isPrimed(), capR.isPrimed()
    )
    CaptureTask(capL, capR, qL, qR).start()

    OSCSenderTask(results, shared_cfg, cfg_lock).start()

    # keep main alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down…")

if __name__ == "__main__":
    main()
