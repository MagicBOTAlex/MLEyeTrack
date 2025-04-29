#!/usr/bin/env python3
import os
import json
import time
import logging

import numpy as np
import cv2
import tensorflow as tf
from pythonosc import udp_client

from mjpeg_streamer import MJPEGVideoCapture

# ----------------------------
# Normalization & OSC Helpers
# ----------------------------
class Norm:
    # Copy these exact values from your TS NormalizationUtils
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
    if v >= 0:
        return v / Norm.maxPosTheta1
    else:
        return v / abs(Norm.maxNegTheta1)

def normalize_theta2(v: float) -> float:
    return v / Norm.maxAbsTheta2

# ----------------------------
# Openness Transform
# ----------------------------
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
# Image Preprocessing & Config
# ----------------------------
def load_and_preprocess(frame: np.ndarray, size=(128,128)) -> tf.Tensor:
    img = cv2.resize(frame, size)
    return tf.convert_to_tensor(img, dtype=tf.float32) / 255.0

def load_config(path="./Settings.json"):
    with open(path, "r") as f:
        return json.load(f)

# ----------------------------
# Main Loop
# ----------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    cfg = load_config()

    # OSC client
    osc_host, osc_port = cfg["vrcOsc"].split(":")
    osc = udp_client.SimpleUDPClient(osc_host, int(osc_port))
    prev_osc = cfg["vrcOsc"]

    # MJPEG streams
    capL = MJPEGVideoCapture(f"http://{cfg['leftEye']}")
    capR = MJPEGVideoCapture(f"http://{cfg['rightEye']}")
    capL.open(); capR.open()
    while not capL.isPrimed() or not capR.isPrimed():
        logging.info("Waiting for initial frames…")
        time.sleep(0.1)

    # Load models
    model_dir = cfg["modelFile"]
    logging.info(f"Loading models from {model_dir}…")
    models = {
        "combined_theta": tf.keras.models.load_model(os.path.join(model_dir, "combined_pitchyaw.h5"), compile=False),
        "combined_open" : tf.keras.models.load_model(os.path.join(model_dir, "combined_openness.h5"), compile=False),
        "left_theta"    : tf.keras.models.load_model(os.path.join(model_dir, "left_pitchyaw.h5"), compile=False),
        "left_open"     : tf.keras.models.load_model(os.path.join(model_dir, "left_openness.h5"), compile=False),
        "right_theta"   : tf.keras.models.load_model(os.path.join(model_dir, "right_pitchyaw.h5"), compile=False),
        "right_open"    : tf.keras.models.load_model(os.path.join(model_dir, "right_openness.h5"), compile=False),
    }
    logging.info("Models ready.")

    # State & blink timestamps
    prev_vals = {
        "native_combined_theta": None,
        "native_indep_theta":    None,
        "native_open":          None,
        "v1_combined_theta":    None,
        "v1_indep_theta":       None,
        "v1_open_comb":         None,
        "v1_open_indep":        None,
        "v2_combined_theta":    None,
        "v2_indep_theta":       None,
        "v2_open_comb":         None,
        "v2_open_indep":        None,
    }
    blink_ts = {"combined":0, "left":0, "right":0}
    prev_ts  = {"left":0, "right":0}
    rate_ms  = cfg["trackingRate"]

    logging.info("Starting main loop. Ctrl+C to stop.")
    try:
        while True:
            now_ms = int(time.time() * 1000)

            # reload config if changed
            new_cfg = load_config()
            if new_cfg["vrcOsc"] != prev_osc:
                osc_host, osc_port = new_cfg["vrcOsc"].split(":")
                osc = udp_client.SimpleUDPClient(osc_host, int(osc_port))
                prev_osc = new_cfg["vrcOsc"]
                logging.info("OSC endpoint updated → %s", prev_osc)
            cfg = new_cfg

            okL, fL = capL.read()
            okR, fR = capR.read()

            # frame timing
            proceed = False
            if cfg["syncedEyeUpdates"] and okL and okR:
                if now_ms - prev_ts["left"] >= rate_ms and now_ms - prev_ts["right"] >= rate_ms:
                    proceed = True
                    prev_ts["left"] = prev_ts["right"] = now_ms
            else:
                if okL and now_ms - prev_ts["left"] >= rate_ms:
                    proceed = True
                    prev_ts["left"] = now_ms
                if okR and now_ms - prev_ts["right"] >= rate_ms:
                    proceed = True
                    prev_ts["right"] = now_ms

            if not proceed:
                time.sleep(0.001)
                continue

            # preprocess
            lt = load_and_preprocess(fL) if okL else tf.zeros((128,128,3))
            rt = load_and_preprocess(fR) if okR else tf.zeros((128,128,3))
            lt_b, rt_b = tf.expand_dims(lt,0), tf.expand_dims(rt,0)

            # openness
            if cfg["activeOpennessTracking"]:
                if not cfg["independentOpenness"] and okL and okR:
                    raw = models["combined_open"].predict([lt_b, rt_b])[0,0]
                    oL = oR = transform_openness(raw, cfg["opennessSliderHandles"])
                else:
                    oL = transform_openness(models["left_open"].predict(lt_b)[0,0], cfg["opennessSliderHandles"]) if okL else None
                    oR = transform_openness(models["right_open"].predict(rt_b)[0,0], cfg["opennessSliderHandles"]) if okR else None
            else:
                oL = oR = None

            # blink‐release delay
            br_s = cfg["blinkReleaseDelayMs"] / 1000.0
            t_now = time.time()
            if oL == 0:    blink_ts["left"] = t_now
            elif t_now <= blink_ts["left"] + br_s:   oL = 0
            if oR == 0:    blink_ts["right"] = t_now
            elif t_now <= blink_ts["right"] + br_s:  oR = 0
            if oL is not None and oR is not None and not cfg["independentOpenness"]:
                if oL == 0:    blink_ts["combined"] = t_now
                elif t_now <= blink_ts["combined"] + br_s:
                    oL = oR = 0

            # gaze‐trust (fallback)
            trust_comb = trust_L = trust_R = 0.75
            if okL and okR and not cfg["independentOpenness"]:
                trust_comb = oL
            else:
                if okL: trust_L = oL
                if okR: trust_R = oR
            if not cfg["eyelidBasedGazeTrust"]:
                trust_comb = trust_L = trust_R = 0.75

            # compute theta with normalization, offset & clamping
            t_comb = tL = tR = None
            if cfg["activeEyeTracking"]:
                off_frac = calculate_offset_fraction(cfg["pitchOffset"])
                # combined
                if not cfg["independentEyes"] and okL and okR:
                    m = models["combined_theta"]
                    inputs = [lt_b, rt_b]
                    if len(m.inputs) == 3:
                        inputs.append(tf.convert_to_tensor([[trust_comb]], dtype=tf.float32))
                    out = m.predict(inputs)[0]
                    raw_p, raw_y = float(out[0]), float(out[1])
                    n1 = normalize_theta1(raw_p)
                    n2 = normalize_theta2(raw_y)
                    y = scale_offset_and_clamp(n1, off_frac, cfg["verticalExaggeration"])
                    x = scale_and_clamp(n2, cfg["horizontalExaggeration"])
                    t_comb = (y, x)
                else:
                    # independent
                    if okL:
                        outL = models["left_theta"].predict(lt_b)[0]
                        rp, ry = float(outL[0]), float(outL[1])
                        n1, n2 = normalize_theta1(rp), normalize_theta2(ry)
                        yL = scale_offset_and_clamp(n1, off_frac, cfg["verticalExaggeration"])
                        xL = scale_and_clamp(n2, cfg["horizontalExaggeration"])
                        tL = (yL, xL)
                    if okR:
                        outR = models["right_theta"].predict(rt_b)[0]
                        rp, ry = float(outR[0]), float(outR[1])
                        n1, n2 = normalize_theta1(rp), normalize_theta2(ry)
                        yR = scale_offset_and_clamp(n1, off_frac, cfg["verticalExaggeration"])
                        xR = scale_and_clamp(n2, cfg["horizontalExaggeration"])
                        tR = (yR, xR)

            # determine mode
            mode = (
                "none" if cfg["trackingForcedOffline"]
                else "native" if cfg["vrcNative"]
                else "v1"    if cfg["vrcftV1"]
                else "v2"    if cfg["vrcftV2"]
                else "none"
            )

            def changed(key, val):
                if prev_vals[key] != val:
                    prev_vals[key] = val
                    return True
                return False

            # ---- NATIVE MODE ----
            if mode == "native":
                # pitch/yaw
                if cfg["activeEyeTracking"]:
                    if cfg["independentEyes"]:
                        if tL and tR and changed("native_indep_theta", (tL, tR)):
                            osc.send_message(
                                "/avatar/eye/native/independent",
                                [tL[0], tL[1], tR[0], tR[1]]
                            )
                    else:
                        if t_comb and changed("native_combined_theta", t_comb):
                            osc.send_message(
                                "/avatar/eye/native/combined",
                                [t_comb[0], t_comb[1]]
                            )
                # openness (always combined)
                if cfg["activeOpennessTracking"] and oL is not None:
                    if changed("native_open", oL):
                        osc.send_message(
                            "/avatar/eye/native/openness",
                            [oL]
                        )

            # ---- VRCFT V1 MODE ----
            elif mode == "v1":
                # pitch/yaw
                if cfg["activeEyeTracking"]:
                    if cfg["independentEyes"]:
                        if tL and tR and changed("v1_indep_theta", (tL, tR)):
                            osc.send_message("/avatar/parameters/LeftEyeX",  [tL[1]])
                            osc.send_message("/avatar/parameters/RightEyeX", [tR[1]])
                            osc.send_message("/avatar/parameters/EyesY",    [-tL[0]])
                    else:
                        if t_comb and changed("v1_combined_theta", t_comb):
                            osc.send_message("/avatar/parameters/LeftEyeX",  [t_comb[1]])
                            osc.send_message("/avatar/parameters/RightEyeX", [t_comb[1]])
                            osc.send_message("/avatar/parameters/EyesY",    [-t_comb[0]])
                # openness
                if cfg["activeOpennessTracking"]:
                    if cfg["independentOpenness"]:
                        if oL is not None and oR is not None and changed("v1_open_indep", (oL, oR)):
                            osc.send_message("/avatar/parameters/LeftEyeLid",  [oL])
                            osc.send_message("/avatar/parameters/RightEyeLid", [oR])
                    else:
                        if oL is not None and changed("v1_open_comb", oL):
                            osc.send_message("/avatar/parameters/CombinedEyeLid", [oL])

            # ---- VRCFT V2 MODE ----
            elif mode == "v2":
                # build prefix
                raw_pfx = cfg["oscPrefix"].strip("/")
                base = f"/avatar/parameters/{raw_pfx}/v2/" if raw_pfx else "/avatar/parameters/v2/"

                # pitch/yaw
                if cfg["activeEyeTracking"]:
                    if cfg["independentEyes"]:
                        if tL and tR and changed("v2_indep_theta", (tL, tR)):
                            ly, lx = tL; ry, rx = tR
                            if cfg["splitOutputY"]:
                                osc.send_message(base + "EyeLeftX",  [lx])
                                osc.send_message(base + "EyeLeftY",  [-ly])
                                osc.send_message(base + "EyeRightX", [rx])
                                osc.send_message(base + "EyeRightY", [-ry])
                            else:
                                osc.send_message(base + "EyeLeftX",  [lx])
                                osc.send_message(base + "EyeRightX", [rx])
                                osc.send_message(base + "EyeY",      [-ly])
                    else:
                        if t_comb and changed("v2_combined_theta", t_comb):
                            y, x = t_comb
                            if cfg["splitOutputY"]:
                                osc.send_message(base + "EyeLeftX",  [x])
                                osc.send_message(base + "EyeLeftY",  [-y])
                                osc.send_message(base + "EyeRightX", [x])
                                osc.send_message(base + "EyeRightY", [-y])
                            else:
                                osc.send_message(base + "EyeLeftX",  [x])
                                osc.send_message(base + "EyeRightX", [x])
                                osc.send_message(base + "EyeY",      [-y])

                # openness
                if cfg["activeOpennessTracking"]:
                    if cfg["independentOpenness"]:
                        if oL is not None and oR is not None and changed("v2_open_indep", (oL, oR)):
                            osc.send_message(base + "EyeLidLeft",  [oL])
                            osc.send_message(base + "EyeLidRight", [oR])
                    else:
                        if oL is not None and changed("v2_open_comb", oL):
                            osc.send_message(base + "EyeLidLeft",  [oL])
                            osc.send_message(base + "EyeLidRight", [oL])

            time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Stopping…")
    finally:
        capL.release(); capR.release()
        logging.info("All streams closed.")

if __name__ == "__main__":
    main()
