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
from pythonosc import udp_client
from mjpeg_streamer import MJPEGVideoCapture

# ----------------------------
# Normalization & OSC Helpers
# ----------------------------
class Norm:
    maxPosTheta1 = 30.0
    maxNegTheta1 = -25.0
    maxAbsTheta2  = 30.0

def clamp(v: float, mn: float = -1.0, mx: float = 1.0) -> float:
    return max(mn, min(mx, v))

def scale_and_clamp(v: float, mul: float) -> float:
    return clamp(v * mul)

def scale_offset_and_clamp(v: float, offset: float, mul: float) -> float:
    return clamp((v - offset) * mul)

def calculate_offset_fraction(pitch_offset: float) -> float:
    if pitch_offset == 0:
        return 0.0
    span = Norm.maxPosTheta1 + abs(Norm.maxNegTheta1)
    return (2 * pitch_offset) / span

def normalize_theta1(v: float) -> float:
    return v / Norm.maxPosTheta1 if v >= 0 else v / abs(Norm.maxNegTheta1)

def normalize_theta2(v: float) -> float:
    return v / Norm.maxAbsTheta2

def transform_openness(val: float, cfg: list) -> float:
    h0, h1, h2, h3 = cfg
    if val < h0:
        return 0.0
    elif val < h1:
        return ((val - h0) / (h1 - h0)) * 0.75
    elif val < h2:
        return 0.75
    elif val < h3:
        return 0.75 + ((val - h2) / (h3 - h2)) * 0.25
    else:
        return 1.0

# ----------------------------
# Config Loader Task
# ----------------------------
class ConfigTask(threading.Thread):
    def __init__(self, path, shared, lock, interval=0.5):
        super().__init__(daemon=True)
        self.path = path
        self.shared = shared
        self.lock = lock
        self.interval = interval
        self._last_mtime = 0

    def run(self):
        while True:
            try:
                mtime = os.path.getmtime(self.path)
                if mtime != self._last_mtime:
                    with open(self.path, 'r') as f:
                        cfg = json.load(f)
                    with self.lock:
                        self.shared.clear()
                        self.shared.update(cfg)
                    self._last_mtime = mtime
                    logging.info("Config reloaded.")
            except FileNotFoundError:
                logging.warning("Settings.json not found.")
            time.sleep(self.interval)

# ----------------------------
# Frame Capture Task
# ----------------------------
class CaptureTask(threading.Thread):
    def __init__(self, capL, capR, queueL, queueR):
        super().__init__(daemon=True)
        self.capL, self.capR = capL, capR
        self.queueL, self.queueR = queueL, queueR

    def run(self):
        while True:
            okL, fL = self.capL.read()
            okR, fR = self.capR.read()
            if okL:
                if self.queueL.full(): self.queueL.get_nowait()
                self.queueL.put(fL)
            if okR:
                if self.queueR.full(): self.queueR.get_nowait()
                self.queueR.put(fR)
            time.sleep(0.001)

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

# ----------------------------
# Post-Process & OSC Sender Task
# ----------------------------
class OSCSenderTask(threading.Thread):
    def __init__(self, result_queue, shared, lock):
        super().__init__(daemon=True)
        self.queue = result_queue
        self.shared = shared
        self.lock = lock
        self.prev = {}
        self.blink_ts = {"left":0, "right":0, "combined":0}

    def run(self):
        osc = None
        while True:
            data = self.queue.get()
            with self.lock:
                cfg = dict(self.shared)
            # OSC client update
            if osc is None or cfg["vrcOsc"] != f"{osc._address}:{osc._port}":
                host, port = cfg["vrcOsc"].split(":")
                osc = udp_client.SimpleUDPClient(host, int(port))
                logging.info("OSC endpoint set to %s:%s", host, port)

            # blink-release logic
            now = time.time()
            for eye in ["left","right"]:
                o = data.get("o"+eye[0].upper())
                if o == 0:
                    self.blink_ts[eye] = now
                elif now <= self.blink_ts[eye] + cfg["blinkReleaseDelayMs"]/1000.0:
                    data["o"+eye[0].upper()] = 0

            if "oL" in data and not cfg.get("independentOpenness",False):
                cb = data["oL"]
                if cb == 0:
                    self.blink_ts["combined"] = now
                elif now <= self.blink_ts["combined"] + cfg["blinkReleaseDelayMs"]/1000.0:
                    data["oL"] = data["oR"] = 0

            # determine mode and send
            mode = ("none" if cfg["trackingForcedOffline"]
                    else "native" if cfg["vrcNative"]
                    else "v1"    if cfg["vrcftV1"]
                    else "v2"    if cfg["vrcftV2"]
                    else "none")

            def send(key, *vals):
                if self.prev.get(key) != tuple(vals):
                    self.prev[key] = tuple(vals)
                    osc.send_message(key, list(vals))

            # NATIVE
            if mode == "native":
                if "t_comb" in data and not cfg.get("independentEyes",False):
                    send("/avatar/eye/native/combined", *data["t_comb"])
                elif "tL" in data and cfg.get("independentEyes",False):
                    yL,xL = data["tL"]; yR,xR = data["tR"]
                    send("/avatar/eye/native/independent", yL, xL, yR, xR)
                if "oL" in data:
                    send("/avatar/eye/native/openness", data["oL"])

            # V1
            elif mode == "v1":
                if "t_comb" in data and not cfg.get("independentEyes",False):
                    y,x = data["t_comb"]
                    send("/avatar/parameters/LeftEyeX", x)
                    send("/avatar/parameters/RightEyeX", x)
                    send("/avatar/parameters/EyesY",    -y)
                elif "tL" in data and cfg.get("independentEyes",False):
                    yL,xL = data["tL"]; yR,xR = data["tR"]
                    send("/avatar/parameters/LeftEyeX", xL)
                    send("/avatar/parameters/RightEyeX", xR)
                    send("/avatar/parameters/EyesY",    -yL)
                if "oL" in data:
                    if cfg.get("independentOpenness",False):
                        send("/avatar/parameters/LeftEyeLid",  data["oL"])
                        send("/avatar/parameters/RightEyeLid", data["oR"])
                    else:
                        send("/avatar/parameters/CombinedEyeLid", data["oL"])

            # V2
            elif mode == "v2":
                pfx = cfg["oscPrefix"].strip("/")
                base = f"/avatar/parameters/{pfx}/v2/" if pfx else "/avatar/parameters/v2/"
                splitY = cfg.get("splitOutputY", False)
                if "t_comb" in data and not cfg.get("independentEyes",False):
                    y,x = data["t_comb"]
                    if splitY:
                        send(base+"EyeLeftX",  x); send(base+"EyeLeftY",  -y)
                        send(base+"EyeRightX", x); send(base+"EyeRightY", -y)
                    else:
                        send(base+"EyeLeftX",  x); send(base+"EyeRightX", x)
                        send(base+"EyeY",      -y)
                elif "tL" in data and cfg.get("independentEyes",False):
                    yL,xL = data["tL"]; yR,xR = data["tR"]
                    if splitY:
                        send(base+"EyeLeftX",  xL); send(base+"EyeLeftY",  -yL)
                        send(base+"EyeRightX", xR); send(base+"EyeRightY", -yR)
                    else:
                        send(base+"EyeLeftX",  xL); send(base+"EyeRightX", xR)
                        send(base+"EyeY",      -yL)
                if "oL" in data:
                    if cfg.get("independentOpenness",False):
                        send(base+"EyeLidLeft",  data["oL"])
                        send(base+"EyeLidRight", data["oR"])
                    else:
                        send(base+"EyeLidLeft",  data["oL"])
                        send(base+"EyeLidRight", data["oL"])

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
