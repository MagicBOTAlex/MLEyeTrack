import requests
import numpy as np
import cv2
import threading
import time

from cameras.ICameraSource import ICameraSource

# Source: https://github.com/Project-Babble/ProjectBabble/pull/105/commits/48938d19d15c177beaa04461b28d7959e1343d53
# License is Project babble's

class MJPEGVideoCapture(ICameraSource):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        
        super().__init__()
    
    def _update(self):
        while self.running:
            try:
                self.stream = self.session.get(self.url, stream=True, timeout=0.5)
                for chunk in self.stream.iter_content(chunk_size=1024):
                    if not self.running:
                        break
                    self.byte_buffer += chunk
                    # Process all available complete frames in the buffer
                    while True:
                        start = self.byte_buffer.find(b'\xff\xd8')  # JPEG start marker
                        end = self.byte_buffer.find(b'\xff\xd9')    # JPEG end marker
                        if start != -1 and end != -1:
                            jpg = self.byte_buffer[start:end+2]
                            self.byte_buffer = self.byte_buffer[end+2:]
                            
                            image = np.frombuffer(jpg, dtype=np.uint8)
                            if image.size != 0:
                                frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    with self.lock:
                                        self.frame = frame  # Always update to the latest frame
                        else:
                            break
            except requests.RequestException:
                # If a network error occurs, wait briefly and retry
                time.sleep(0.1)
                continue

    


if __name__ == "__main__":
    cap = MJPEGVideoCapture("http://openiristracker.local")
    cap.open()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imshow("MJPEG Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
