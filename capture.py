import logging
import threading
import time
from helpers import *

# ----------------------------
# Setup basic logging once
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ----------------------------
# Frame Capture Task
# ----------------------------
class CaptureTask(threading.Thread):
    def __init__(self, capL, capR, queueL, queueR):
        super().__init__(daemon=True)
        self.capL, self.capR = capL, capR
        self.queueL, self.queueR = queueL, queueR

        # Counters for left & right
        self.countL = 0
        self.countR = 0
        # Single timer for logging both together
        self.last_log_time = time.time()

    def run(self):
        while True:
            now = time.time()

            # --- Left camera ---
            okL, fL = self.capL.read()
            if okL:
                if self.queueL.full():
                    self.queueL.get_nowait()
                self.queueL.put(fL)
                self.countL += 1

            # --- Right camera ---
            okR, fR = self.capR.read()
            if okR:
                if self.queueR.full():
                    self.queueR.get_nowait()
                self.queueR.put(fR)
                self.countR += 1

            # Once a second, log both FPS in one line
            elapsed = now - self.last_log_time
            if elapsed >= 1.0:
                fpsL = self.countL / elapsed
                fpsR = self.countR / elapsed
                logging.info(f"Left: {fpsL:.2f} fps, Right: {fpsR:.2f} fps")
                # reset counters & timer
                self.countL = 0
                self.countR = 0
                self.last_log_time = now

            time.sleep(0.001)
