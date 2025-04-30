import logging
import threading
import time
from helpers import *

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
                if self.queueL.full(): 
                    self.queueL.get_nowait()
                    
                # logging.info("Got left frame")
                self.queueL.put(fL)
            if okR:
                if self.queueR.full(): 
                    self.queueR.get_nowait()
                    
                # logging.info("Got right frame")
                self.queueR.put(fR)
            time.sleep(0.001)