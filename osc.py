import threading
import time
from helpers import *
from pythonosc import udp_client
import logging

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