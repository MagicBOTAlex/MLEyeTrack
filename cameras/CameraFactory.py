import re
import logging

from .ICameraSource import ICameraSource
from .MJPEGVideoCapture import MJPEGVideoCapture
from .SerialCameraCapture import SerialCamera
from .SystemCameraCapture import SystemCamera

# Sorry for the (non-OOP) Python devs. Factory time!
class CameraFactory:
    @staticmethod
    def get_camera_from_string_type(sourceName: str) -> ICameraSource:
        source = str(sourceName).strip()

        # 1) HTTP(S) URL
        if re.match(r'^(https?://)', source, re.IGNORECASE):
            logging.log(logging.INFO, f"MJPEG camera selected: {source}")
            return MJPEGVideoCapture(source)

        # 2) Plain IPv4 address (assume HTTP)
        if re.match(r'^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$', source):
            url = f"http://{source}"
            logging.log(logging.INFO, f"Plain IP address detected, using MJPEG camera: {url}")
            return MJPEGVideoCapture(url)

        # 3) Serial ports on Windows (COM), macOS (/dev/cu) or Linux (/dev/tty)
        if source.lower().startswith(("com", "/dev/cu", "/dev/tty")):
            logging.log(logging.INFO, f"Serial camera selected: {source}")
            return SerialCamera(source)

        # 4) Fallback to system camera (e.g. integer index or device path)
        logging.log(logging.INFO, f"System camera selected: {source}")
        return SystemCamera(source)