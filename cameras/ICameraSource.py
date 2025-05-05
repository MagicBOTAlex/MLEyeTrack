# Called ICamera, but it's going to be abstract

from abc import abstractmethod
import threading

import requests

# License is Project babble's. (because this is derived)
# Also derived from: https://github.com/MagicBOTAlex/EyeTrackVR/blob/v2.0-beta-feature-branch/EyeTrackApp/Camera/ICameraSource.py

class ICameraSource:
    def __init__(self):
        self.stream = None
        self.byte_buffer = b""
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
    
    def read(self):
        if self.frame is not None:
            frame_copy = self.frame.copy()
            self.frame = None
            return True, frame_copy
        else:
            return False, None
        
    @abstractmethod
    def _update(self):
        pass
        
    def open(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def isOpened(self):
       return self.running
    
    def isPrimed(self):
        if self.frame is not None:
            return True
        else: return False

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.stream = None
        self.frame = None
        self.byte_buffer = b""
        self.session.close()