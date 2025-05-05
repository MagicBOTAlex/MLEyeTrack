import os
import sys
import json
import time
import logging
import threading
import subprocess
import shutil
from queue import Queue, Empty

import tensorflow as tf

import numpy as np
import cv2
import onnxruntime as ort
from pythonosc import udp_client
import tf2onnx
from MJPEGVideoCapture import MJPEGVideoCapture
from helpers import (
    transform_openness,
    calculate_offset_fraction,
    normalize_theta1,
    normalize_theta2,
    scale_offset_and_clamp,
    scale_and_clamp,
)


import onnxruntime as ort

def get_onnx_providers():
    """
    Return a list of ONNX Runtime providers, preferring GPU if available.
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        # You may also configure session options here (e.g. memory limits).
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]
    
# --------------------------------
# ONNX conversion helper
# --------------------------------
def ensure_onnx(h5_path: str, onnx_path: str, opset: int = 13):
    # Only convert if ONNX is missing or stale
    if (not os.path.exists(onnx_path)
        or os.path.getmtime(h5_path) > os.path.getmtime(onnx_path)):

        logging.info(f"Converting {os.path.basename(h5_path)} → ONNX…")
        model = tf.keras.models.load_model(h5_path, compile=False)

        # 1) Patch output_names if needed
        if not hasattr(model, "output_names"):
            model.output_names = [
                tensor.name.split(":")[0]
                for tensor in model.outputs
            ]

        # 2) Build an input_signature
        input_signature = [
            tf.TensorSpec(inp.shape, inp.dtype, name=inp.name.split(":")[0])
            for inp in model.inputs
        ]

        # 3) Convert in-process
        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
            output_path=onnx_path
        )
        logging.info(f"  ✓ Wrote ONNX to {onnx_path}")
        
# --------------------------------
# Load ONNX sessions
# --------------------------------
def load_models(model_dir):
    specs = {
        "combined_theta": "combined_pitchyaw.h5",
        "combined_open" : "combined_openness.h5",
        "left_theta"    : "left_pitchyaw.h5",
        "left_open"     : "left_openness.h5",
        "right_theta"   : "right_pitchyaw.h5",
        "right_open"    : "right_openness.h5",
    }
    sessions = {}
    providers = get_onnx_providers()

    for key, fname in specs.items():
        h5_path   = os.path.join(model_dir, fname)
        onnx_path = os.path.splitext(h5_path)[0] + ".onnx"
        ensure_onnx(h5_path, onnx_path)

        # Create the session with GPU if available
        sess = ort.InferenceSession(
            onnx_path,
            providers=providers
        )
        print("Using: " + sess.get_providers()[0])
        sessions[key] = sess

    return sessions

# --------------------------------
# Inference Task using ONNX
# --------------------------------
class InferenceTask(threading.Thread):
    def __init__(self, cfg, queueL, queueR, result_queue, shared, lock):
        super().__init__(daemon=True)
        self.models = load_models(cfg["modelFile"])
        self.queueL, self.queueR = queueL, queueR
        self.result_queue = result_queue
        self.shared = shared
        self.lock = lock
        self.fL = None
        self.fR = None
        
        self.last_theta = {"tL": (0.0, 0.0), "tR": (0.0, 0.0)}
        self.last_open  = {"oL": 0.0,        "oR": 0.0}

    def preprocess(self, frame):
        img = cv2.resize(frame, (128, 128))
        return img.astype(np.float32) / 255.0

    def run(self):
        infer_count = 0
        start_time  = time.perf_counter()

        while True:
            # ────────────────────── 1. grab the latest frames ──────────────────
            try:
                self.fL = self.queueL.get(timeout=0.1)
            except Empty:
                self.fL = None

            try:
                self.fR = self.queueR.get(timeout=0.1)
            except Empty:
                self.fR = None

            # if absolutely nothing new, take a tiny nap and loop
            if self.fL is None and self.fR is None:
                time.sleep(0.01)
                continue

            lt_np = (np.expand_dims(self.preprocess(self.fL), 0)
                    if self.fL is not None else None)
            rt_np = (np.expand_dims(self.preprocess(self.fR), 0)
                    if self.fR is not None else None)

            # take a snapshot of shared settings
            with self.lock:
                cfg = dict(self.shared)

            outputs = {}

            # ────────────────────── 2. openness inference ──────────────────────
            if cfg.get("activeOpennessTracking", False):
                handles = cfg["opennessSliderHandles"]

                if not cfg.get("independentOpenness", False):
                    # combined model needs *both* eyes
                    if lt_np is not None and rt_np is not None:
                        sess   = self.models["combined_open"]
                        i0, i1 = (t.name for t in sess.get_inputs())
                        raw    = sess.run(None, {i0: lt_np, i1: rt_np})[0].item()
                        o      = transform_openness(raw, handles)
                        outputs["oL"] = outputs["oR"] = o
                else:
                    if lt_np is not None:
                        sess = self.models["left_open"]
                        raw  = sess.run(None, {sess.get_inputs()[0].name: lt_np})[0].item()
                        outputs["oL"] = transform_openness(raw, handles)
                    if rt_np is not None:
                        sess = self.models["right_open"]
                        raw  = sess.run(None, {sess.get_inputs()[0].name: rt_np})[0].item()
                        outputs["oR"] = transform_openness(raw, handles)

            # ────────────────────── 3. pitch / yaw inference ────────────────────
            if cfg.get("activeEyeTracking", False):
                off_frac = calculate_offset_fraction(cfg["pitchOffset"])
                hor, ver = cfg["horizontalExaggeration"], cfg["verticalExaggeration"]

                if not cfg.get("independentEyes", False):
                    # combined model needs *both* eyes
                    if lt_np is not None and rt_np is not None:
                        sess            = self.models["combined_theta"]
                        i0, i1, *opt    = sess.get_inputs()
                        feed            = {i0.name: lt_np, i1.name: rt_np}
                        if opt:         # optional “head pose” scalar
                            feed[opt[0].name] = np.array([[0.75]], np.float32)
                        raw_p, raw_y    = sess.run(None, feed)[0][0]
                        n1, n2          = normalize_theta1(raw_p), normalize_theta2(raw_y)
                        outputs["t_comb"] = (
                            scale_offset_and_clamp(n1, off_frac, ver),
                            scale_and_clamp(n2, hor)
                        )
                else:
                    if lt_np is not None:
                        sess = self.models["left_theta"]
                        p, y = sess.run(None, {sess.get_inputs()[0].name: lt_np})[0][0]
                        n1, n2 = normalize_theta1(p), normalize_theta2(y)
                        outputs["tL"] = (
                            scale_offset_and_clamp(n1, off_frac, ver),
                            scale_and_clamp(n2, hor)
                        )
                    if rt_np is not None:
                        sess = self.models["right_theta"]
                        p, y = sess.run(None, {sess.get_inputs()[0].name: rt_np})[0][0]
                        n1, n2 = normalize_theta1(p), normalize_theta2(y)
                        outputs["tR"] = (
                            scale_offset_and_clamp(n1, off_frac, ver),
                            scale_and_clamp(n2, hor)
                        )

            # ────────────────────── 4. fill missing keys with last-seen values ─
            for k in ("tL", "tR"):
                if k not in outputs:
                    outputs[k] = self.last_theta[k]
            for k in ("oL", "oR"):
                if k not in outputs:
                    outputs[k] = self.last_open[k]

            # update the “last-seen” caches
            self.last_theta.update({k: outputs[k] for k in ("tL", "tR")})
            self.last_open .update({k: outputs[k] for k in ("oL", "oR")})

            # ────────────────────── 5. hand results to the consumer ─────────────
            self.result_queue.put(outputs)
            infer_count += 1

            # ────────────────────── 6. rate logging and sleep ───────────────────
            now      = time.perf_counter()
            elapsed  = now - start_time
            if elapsed >= 1.0:
                logging.info(f"Inference rate: {infer_count / elapsed:.2f} updates/s")
                infer_count = 0
                start_time  = now

            # with self.lock:
            #     interval = self.shared.get("trackingRate", 50) / 1000.0
            # time.sleep(interval)
