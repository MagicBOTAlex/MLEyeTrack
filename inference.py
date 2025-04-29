import os
import sys
import json
import time
import logging
import threading
import subprocess
import shutil
from queue import Queue, Empty

import numpy as np
import cv2
import onnxruntime as ort
from pythonosc import udp_client
from mjpeg_streamer import MJPEGVideoCapture
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
def ensure_onnx(h5_path: str, onnx_path: str, opset=13):
    """
    1) If .onnx is missing or older than .h5,
       a) load .h5, save as SavedModel,
       b) run tf2onnx on that SavedModel.
    """
    if (not os.path.exists(onnx_path)
        or os.path.getmtime(h5_path) > os.path.getmtime(onnx_path)):

        logging.info(f"Converting {os.path.basename(h5_path)} → ONNX…")

        # a) Load your Keras .h5
        import tensorflow as tf  # only here
        model = tf.keras.models.load_model(h5_path, compile=False)

        # b) Dump to a temporary SavedModel folder
        sm_dir = onnx_path + "_savedmodel"
        if os.path.isdir(sm_dir):
            shutil.rmtree(sm_dir)
        tf.saved_model.save(model, sm_dir)
        logging.info(f"  • SavedModel written to {sm_dir}")

        # c) Call the tf2onnx CLI
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", sm_dir,
            "--output",      onnx_path,
            "--opset",      str(opset)
        ]
        try:
            subprocess.check_call(cmd)
            logging.info(f"  ✓ Wrote ONNX to {onnx_path}")
        except subprocess.CalledProcessError as e:
            logging.error("ONNX conversion failed.")
            logging.error(e)
            raise

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

    def preprocess(self, frame):
        img = cv2.resize(frame, (128, 128))
        return img.astype(np.float32) / 255.0

    def run(self):
        infer_count = 0
        start_time = time.perf_counter()
        
        while True:
            try:
                fL = self.queueL.get(timeout=1)
                fR = self.queueR.get(timeout=1)
            except Empty:
                continue

            lt_np = np.expand_dims(self.preprocess(fL), axis=0)
            rt_np = np.expand_dims(self.preprocess(fR), axis=0)

            with self.lock:
                cfg = dict(self.shared)
            outputs = {}

            # Openness
            if cfg.get("activeOpennessTracking", False):
                handles = cfg["opennessSliderHandles"]
                if not cfg.get("independentOpenness", False):
                    sess = self.models["combined_open"]
                    inp_names = [i.name for i in sess.get_inputs()]
                    raw = sess.run(
                        None,
                        { inp_names[0]: lt_np, inp_names[1]: rt_np }
                    )[0].item()
                    o = transform_openness(raw, handles)
                    outputs['oL'] = outputs['oR'] = o
                else:
                    for eye, key in (("oL", "left_open"), ("oR", "right_open")):
                        sess = self.models[key]
                        inp = sess.get_inputs()[0].name
                        raw = sess.run(None, {inp: (lt_np if eye=="oL" else rt_np)})[0].item()
                        outputs[eye] = transform_openness(raw, handles)

            # Pitch/Yaw
            if cfg.get("activeEyeTracking", False):
                off_frac = calculate_offset_fraction(cfg["pitchOffset"])
                hor, ver = cfg["horizontalExaggeration"], cfg["verticalExaggeration"]

                if not cfg.get("independentEyes", False):
                    sess = self.models["combined_theta"]
                    inp_defs = sess.get_inputs()
                    feed = {
                        inp_defs[0].name: lt_np,
                        inp_defs[1].name: rt_np,
                    }
                    # handle optional third input
                    if len(inp_defs) == 3:
                        feed[inp_defs[2].name] = np.array([[0.75]], dtype=np.float32)

                    raw_p, raw_y = sess.run(None, feed)[0][0]
                    n1, n2 = normalize_theta1(raw_p), normalize_theta2(raw_y)
                    outputs['t_comb'] = (
                        scale_offset_and_clamp(n1, off_frac, ver),
                        scale_and_clamp(n2, hor)
                    )
                else:
                    for side in ("left", "right"):
                        sess = self.models[f"{side}_theta"]
                        inp = sess.get_inputs()[0].name
                        p, y = sess.run(None, {inp: (lt_np if side=="left" else rt_np)})[0][0]
                        n1, n2 = normalize_theta1(p), normalize_theta2(y)
                        key = 'tL' if side=="left" else 'tR'
                        outputs[key] = (
                            scale_offset_and_clamp(n1, off_frac, ver),
                            scale_and_clamp(n2, hor)
                        )

            self.result_queue.put(outputs)
            infer_count += 1

            # every second, log the rate and reset
            now = time.perf_counter()
            elapsed = now - start_time
            if elapsed >= 1.0:
                rate = infer_count / elapsed
                logging.info(f"Inference rate: {rate:.2f} updates/s")
                infer_count = 0
                start_time = now

            with self.lock:
                interval = self.shared.get("trackingRate", 50) / 1000.0
            time.sleep(interval)
