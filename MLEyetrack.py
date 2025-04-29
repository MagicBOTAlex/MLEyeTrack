#!/usr/bin/env python3
import os
import json
import time
import threading
import logging

import numpy as np
import cv2
import tensorflow as tf
from pythonosc import udp_client

from mjpeg_streamer import MJPEGVideoCapture

# ----------------------------
# Helpers
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

def load_and_preprocess(frame: np.ndarray, size=(128,128)) -> tf.Tensor:
    img = cv2.resize(frame, size)
    return tf.convert_to_tensor(img, dtype=tf.float32) / 255.0

def load_config(path="./Settings.json"):
    with open(path, "r") as f:
        return json.load(f)

# ----------------------------
# Main
# ----------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    cfg = load_config()
    # OSC client state
    osc_host, osc_port = cfg["vrcOsc"].split(":")
    osc = udp_client.SimpleUDPClient(osc_host, int(osc_port))
    prev_osc_endpoint = cfg["vrcOsc"]

    # Open MJPEG streams
    capL = MJPEGVideoCapture(f"http://{cfg['leftEye']}")
    capR = MJPEGVideoCapture(f"http://{cfg['rightEye']}")
    capL.open(); capR.open()

    # wait for first frame
    while not capL.isPrimed() or not capR.isPrimed():
        logging.info("Waiting for initial frames...")
        time.sleep(0.1)

    # Load TF models
    model_dir = cfg["modelFile"]
    logging.info(f"Loading models from {model_dir}...")
    models = {
        "combined_theta": tf.keras.models.load_model(os.path.join(model_dir, "combined_pitchyaw.h5"), compile=False),
        "combined_open" : tf.keras.models.load_model(os.path.join(model_dir, "combined_openness.h5"), compile=False),
        "left_theta"    : tf.keras.models.load_model(os.path.join(model_dir, "left_pitchyaw.h5"), compile=False),
        "left_open"     : tf.keras.models.load_model(os.path.join(model_dir, "left_openness.h5"), compile=False),
        "right_theta"   : tf.keras.models.load_model(os.path.join(model_dir, "right_pitchyaw.h5"), compile=False),
        "right_open"    : tf.keras.models.load_model(os.path.join(model_dir, "right_openness.h5"), compile=False),
    }
    logging.info("Models ready.")

    # State for change-detection & blink-delay
    prev_vals = {
        "native_combined_theta": None,
        "native_indep_theta"   : None,
        "native_open"          : None,
        "v1_combined_theta"    : None,
        "v1_indep_theta"       : None,
        "v1_open_comb"         : None,
        "v1_open_indep"        : None,
        "v2_combined_theta"    : None,
        "v2_indep_theta"       : None,
        "v2_open_comb"         : None,
        "v2_open_indep"        : None,
    }
    blink_ts = {"combined":0, "left":0, "right":0}

    # Frame-timing
    rate_ms = cfg["trackingRate"]
    prev_ts = {"left":0, "right":0}

    logging.info("Starting main loop. Ctrl+C to stop.")
    try:
        while True:
            now = int(time.time()*1000)

            # reload config if changed on disk
            new_cfg = load_config()
            if new_cfg["vrcOsc"] != prev_osc_endpoint:
                osc_host, osc_port = new_cfg["vrcOsc"].split(":")
                osc = udp_client.SimpleUDPClient(osc_host, int(osc_port))
                prev_osc_endpoint = new_cfg["vrcOsc"]
                logging.info("OSC endpoint updated → %s", prev_osc_endpoint)
            cfg = new_cfg

            # grab frames
            okL, fL = capL.read()
            okR, fR = capR.read()

            # check timing
            proceed = False
            if cfg["syncedEyeUpdates"] and okL and okR:
                if now - prev_ts["left"] >= rate_ms and now - prev_ts["right"] >= rate_ms:
                    proceed = True
                    prev_ts["left"] = prev_ts["right"] = now
            else:
                if okL and now - prev_ts["left"] >= rate_ms:
                    proceed = True
                    prev_ts["left"] = now
                if okR and now - prev_ts["right"] >= rate_ms:
                    proceed = True
                    prev_ts["right"] = now

            if not proceed:
                time.sleep(0.001)
                continue

            # preprocess
            lt = load_and_preprocess(fL) if okL else tf.zeros((128,128,3))
            rt = load_and_preprocess(fR) if okR else tf.zeros((128,128,3))
            lt_b = tf.expand_dims(lt,0); rt_b = tf.expand_dims(rt,0)

            # Openness
            if not cfg["activeOpennessTracking"]:
                oL = oR = None
            else:
                if not cfg["independentOpenness"] and okL and okR:
                    raw = models["combined_open"].predict([lt_b, rt_b])[0,0]
                    comb = transform_openness(raw, cfg["opennessSliderHandles"])
                    oL = oR = comb
                else:
                    oL = transform_openness(models["left_open"].predict(lt_b)[0,0], cfg["opennessSliderHandles"]) if okL else None
                    oR = transform_openness(models["right_open"].predict(rt_b)[0,0], cfg["opennessSliderHandles"]) if okR else None

            # apply blink-release delay
            br = cfg["blinkReleaseDelayMs"]
            now_s = time.time()
            if oL == 0: blink_ts["left"] = now_s
            elif now_s <= blink_ts["left"] + br/1000: oL = 0
            if oR == 0: blink_ts["right"] = now_s
            elif now_s <= blink_ts["right"] + br/1000: oR = 0
            if oL is not None and oR is not None and not cfg["independentOpenness"]:
                # combined blink
                if oL == 0: blink_ts["combined"] = now_s
                elif now_s <= blink_ts["combined"] + br/1000:
                    oL = oR = 0

            # Gaze-trust defaults
            trust_comb = trust_L = trust_R = 0.75
            if okL and okR and not cfg["independentOpenness"]:
                trust_comb = oL  # use combined
            else:
                if okL: trust_L = oL
                if okR: trust_R = oR
            if not cfg["eyelidBasedGazeTrust"]:
                trust_comb = trust_L = trust_R = 0.75

            # Theta
            t_comb = None
            if not cfg["activeEyeTracking"]:
                pass
            elif not cfg["independentEyes"] and okL and okR:
                # some models accept a third input for trust
                m = models["combined_theta"]
                inputs = [lt_b, rt_b]
                if len(m.inputs) == 3:
                    inputs.append(tf.convert_to_tensor([[trust_comb]], dtype=tf.float32))
                out = m.predict(inputs)[0]
                pitch, yaw = float(out[0]), float(out[1])
                # apply offset & exaggeration
                pitch = pitch + cfg["pitchOffset"]
                yaw   = yaw * cfg["verticalExaggeration"], yaw * cfg["horizontalExaggeration"]
                t_comb = (pitch, yaw)
            else:
                # independent
                if okL:
                    out = models["left_theta"].predict(lt_b)[0]
                    lp, ly = float(out[0]), float(out[1])
                    lp = lp + cfg["pitchOffset"]
                    ly = ly * cfg["verticalExaggeration"], ly * cfg["horizontalExaggeration"]
                    tL = (lp, ly)
                else:
                    tL = None
                if okR:
                    out = models["right_theta"].predict(rt_b)[0]
                    rp, ry = float(out[0]), float(out[1])
                    rp = rp + cfg["pitchOffset"]
                    ry = ry * cfg["verticalExaggeration"], ry * cfg["horizontalExaggeration"]
                    tR = (rp, ry)
                else:
                    tR = None

            # --------------------------------
            # Now: send OSC based on mode
            # --------------------------------
            mode = ("none" if cfg["trackingForcedOffline"]
                    else "native" if cfg["vrcNative"]
                    else "v1"    if cfg["vrcftV1"]
                    else "v2"    if cfg["vrcftV2"]
                    else "none")

            def changed(key, val):
                prev = prev_vals[key]
                if prev != val:
                    prev_vals[key] = val
                    return True
                return False

            # Native
            if mode == "native":
                if t_comb and changed("native_combined_theta", t_comb):
                    osc.send_message("/native/pitchyaw", list(t_comb))
                if oL is not None and changed("native_open", oL):
                    osc.send_message("/native/openness", oL)

            # V1
            elif mode == "v1":
                if not cfg["independentEyes"] and t_comb and changed("v1_combined_theta", t_comb):
                    osc.send_message("/vrcftV1/combined/pitchyaw", list(t_comb))
                elif cfg["independentEyes"]:
                    if tL and changed("v1_indep_theta", tL):
                        osc.send_message("/vrcftV1/left/pitchyaw", list(tL))
                    if tR and changed("v1_indep_theta", tR):
                        osc.send_message("/vrcftV1/right/pitchyaw", list(tR))
                if cfg["independentOpenness"]:
                    v = (oL, oR)
                    if changed("v1_open_indep", v):
                        osc.send_message("/vrcftV1/left/openness", oL)
                        osc.send_message("/vrcftV1/right/openness", oR)
                else:
                    if oL is not None and changed("v1_open_comb", oL):
                        osc.send_message("/vrcftV1/combined/openness", oL)

            # V2
            elif mode == "v2":
                pre = cfg["oscPrefix"]
                sx = cfg["splitOutputY"]
                # pitch/yaw
                if not cfg["independentEyes"] and t_comb and changed("v2_combined_theta", t_comb):
                    if sx:
                        osc.send_message(f"{pre}combined/x", t_comb[0])
                        osc.send_message(f"{pre}combined/y", t_comb[1])
                    else:
                        osc.send_message(f"{pre}combined", list(t_comb))
                else:
                    if tL and changed("v2_indep_theta", tL):
                        if sx:
                            osc.send_message(f"{pre}left/x",  tL[0])
                            osc.send_message(f"{pre}left/y",  tL[1])
                        else:
                            osc.send_message(f"{pre}left", list(tL))
                    if tR and changed("v2_indep_theta", tR):
                        if sx:
                            osc.send_message(f"{pre}right/x", tR[0])
                            osc.send_message(f"{pre}right/y", tR[1])
                        else:
                            osc.send_message(f"{pre}right", list(tR))
                # openness
                if cfg["independentOpenness"]:
                    v = (oL, oR)
                    if changed("v2_open_indep", v):
                        osc.send_message(f"{pre}left/openness",  oL)
                        osc.send_message(f"{pre}right/openness", oR)
                else:
                    if oL is not None and changed("v2_open_comb", oL):
                        osc.send_message(f"{pre}openness", oL)

            # tiny sleep
            time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Stopping…")
    finally:
        capL.release()
        capR.release()
        logging.info("All streams closed.")

if __name__ == "__main__":
    main()
