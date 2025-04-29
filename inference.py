import os
import json
import time
import logging
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import tensorflow as tf
from pythonosc import udp_client
from mjpeg_streamer import MJPEGVideoCapture
from helpers import *

# ----------------------------
# Inference Task
# ----------------------------
class InferenceTask(threading.Thread):
    def __init__(self, models, queueL, queueR, result_queue, shared, lock):
        super().__init__(daemon=True)
        self.models = models
        self.queueL, self.queueR = queueL, queueR
        self.result_queue = result_queue
        self.shared = shared
        self.lock = lock

    def preprocess(self, frame):
        img = cv2.resize(frame, (128,128))
        return tf.convert_to_tensor(img, dtype=tf.float32) / 255.0

    def run(self):
        while True:
            try:
                fL = self.queueL.get(timeout=1)
                fR = self.queueR.get(timeout=1)
            except Empty:
                continue

            lt = tf.expand_dims(self.preprocess(fL), 0)
            rt = tf.expand_dims(self.preprocess(fR), 0)

            with self.lock:
                cfg = dict(self.shared)
            outputs = {}

            # openness
            if cfg.get("activeOpennessTracking", False):
                handles = cfg["opennessSliderHandles"]
                if not cfg.get("independentOpenness", False):
                    raw = self.models["combined_open"].predict([lt, rt])[0,0]
                    o = transform_openness(raw, handles)
                    outputs['oL'] = outputs['oR'] = o
                else:
                    outputs['oL'] = transform_openness(self.models["left_open"].predict(lt)[0,0], handles)
                    outputs['oR'] = transform_openness(self.models["right_open"].predict(rt)[0,0], handles)

            # pitch/yaw
            if cfg.get("activeEyeTracking", False):
                off_frac = calculate_offset_fraction(cfg["pitchOffset"])
                hor, ver = cfg["horizontalExaggeration"], cfg["verticalExaggeration"]
                if not cfg.get("independentEyes", False):
                    m = self.models["combined_theta"]
                    inp = [lt, rt]
                    if len(m.inputs) == 3:
                        inp.append(tf.convert_to_tensor([[0.75]], dtype=tf.float32))
                    raw_p, raw_y = m.predict(inp)[0]
                    n1, n2 = normalize_theta1(raw_p), normalize_theta2(raw_y)
                    outputs['t_comb'] = (
                        scale_offset_and_clamp(n1, off_frac, ver),
                        scale_and_clamp(n2, hor)
                    )
                else:
                    pL, yL = self.models["left_theta"].predict(lt)[0]
                    pR, yR = self.models["right_theta"].predict(rt)[0]
                    n1L, n2L = normalize_theta1(pL), normalize_theta2(yL)
                    n1R, n2R = normalize_theta1(pR), normalize_theta2(yR)
                    outputs['tL'] = (scale_offset_and_clamp(n1L, off_frac, ver), scale_and_clamp(n2L, hor))
                    outputs['tR'] = (scale_offset_and_clamp(n1R, off_frac, ver), scale_and_clamp(n2R, hor))

            self.result_queue.put(outputs)
            with self.lock:
                rate = self.shared.get("trackingRate", 50) / 1000.0
            time.sleep(rate)