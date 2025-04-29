#!/usr/bin/env python3
import os
import json
import time
import logging
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import tensorflow as tf
from inference import InferenceTask
from osc import OSCSenderTask
from mjpeg_streamer import MJPEGVideoCapture
from helpers import *
from config import ConfigTask
from capture import CaptureTask

# ----------------------------
# Main
# ----------------------------
def load_models(model_dir):
    return {
        "combined_theta": tf.keras.models.load_model(os.path.join(model_dir, "combined_pitchyaw.h5"), compile=False),
        "combined_open" : tf.keras.models.load_model(os.path.join(model_dir, "combined_openness.h5"), compile=False),
        "left_theta"    : tf.keras.models.load_model(os.path.join(model_dir, "left_pitchyaw.h5"), compile=False),
        "left_open"     : tf.keras.models.load_model(os.path.join(model_dir, "left_openness.h5"), compile=False),
        "right_theta"   : tf.keras.models.load_model(os.path.join(model_dir, "right_pitchyaw.h5"), compile=False),
        "right_open"    : tf.keras.models.load_model(os.path.join(model_dir, "right_openness.h5"), compile=False),
    }

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

    # setup camera and models once
    with cfg_lock:
        cfg = dict(shared_cfg)
    capL = MJPEGVideoCapture(f"http://{cfg['leftEye']}"); capL.open()
    capR = MJPEGVideoCapture(f"http://{cfg['rightEye']}"); capR.open()
    logging.info("Waiting for at least one camera to become ready…")
    while not (capL.isPrimed() or capR.isPrimed()):
        time.sleep(0.1)
    logging.info(
        "Camera ready. leftPrimed=%s, rightPrimed=%s",
        capL.isPrimed(), capR.isPrimed()
    )
    models = load_models(cfg["modelFile"])
    logging.info("Models loaded and cameras ready.")

    # queues
    qL, qR = Queue(maxsize=1), Queue(maxsize=1)
    results = Queue(maxsize=5)

    # start tasks
    CaptureTask(capL, capR, qL, qR).start()
    InferenceTask(models, qL, qR, results, shared_cfg, cfg_lock).start()
    OSCSenderTask(results, shared_cfg, cfg_lock).start()

    # keep main alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down…")

if __name__ == "__main__":
    main()
