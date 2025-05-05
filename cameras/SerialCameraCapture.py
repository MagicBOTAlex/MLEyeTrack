import serial
import cv2
import numpy as np

# Derived from: https://github.com/MagicBOTAlex/EyeTrackVR/blob/v2.0-beta-feature-branch/EyeTrackApp/Camera/SerialCamera.py
# Not babble license
# Fed directly into chatGPT

from cameras.ICameraSource import ICameraSource
class SerialCamera(ICameraSource):
    def __init__(self, port, baudrate=3000000):
        """
        :param port: Serial port (e.g. "COM3" or "/dev/ttyUSB0")
        :param baudrate: Baud rate for the ETVR camera
        """
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.conn = None
        self.buffer = b""

        # ETVR headers
        self.ETVR_HEADER      = b"\xff\xa0"
        self.ETVR_HEADER_FRAME= b"\xff\xa1"
        self.ETVR_HEADER_LEN  = 6

    def _update(self):
        # 1) open serial port
        try:
            self.conn = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
        except Exception as e:
            print(f"[ERROR] Cannot open {self.port}: {e}")
            return

        # 2) read loop
        while self.running:
            data = self.conn.read(2048)
            if not data:
                continue
            self.buffer += data

            # find the start of a frame packet
            idx = self.buffer.find(self.ETVR_HEADER + self.ETVR_HEADER_FRAME)
            if idx < 0:
                # keep only the tail in case header spans reads
                if len(self.buffer) > self.ETVR_HEADER_LEN:
                    self.buffer = self.buffer[-self.ETVR_HEADER_LEN:]
                continue

            # need at least 6 bytes to read packet-size
            if len(self.buffer) < idx + 6:
                continue

            # packet-size is two bytes at offset 4..6
            length = int.from_bytes(self.buffer[idx+4:idx+6], "little")
            frame_end = idx + self.ETVR_HEADER_LEN + length

            # wait until the full JPEG arrives
            if len(self.buffer) < frame_end:
                continue

            # extract and drop it from buffer
            jpeg = self.buffer[idx + self.ETVR_HEADER_LEN : frame_end]
            self.buffer = self.buffer[frame_end:]

            # decode JPEG into BGR image
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                # corrupted, skip
                continue

            # store latest frame
            with self.lock:
                self.frame = frame

        # 3) clean up
        self.conn.close()
