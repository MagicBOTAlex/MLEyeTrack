import json
import logging
import os
import threading
import time
from helpers import *

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