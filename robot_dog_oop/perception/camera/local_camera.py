# perception/camera/local_camera.py
import cv2
import time
import os
from typing import Optional
import numpy as np

from .base import ICameraSource


class LocalCamera(ICameraSource):
    """本地USB摄像头"""

    def __init__(
        self,
        cam_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        save_dir: str = "data"
    ):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.last_time = time.time()
        self._fps_value = 0.0

        self.recording = False
        self.video_writer = None

        self.image_dir = os.path.join(save_dir, "images")
        self.video_dir = os.path.join(save_dir, "videos")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None

        now = time.time()
        self._fps_value = 1.0 / (now - self.last_time)
        self.last_time = now

        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

        return frame

    def release(self):
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def is_opened(self) -> bool:
        return self.cap.is_opened()

    @property
    def fps(self) -> float:
        return self._fps_value

    @property
    def width(self) -> int:
        return self.width

    @property
    def height(self) -> int:
        return self.height

    def start_record(self):
        """开始录像"""
        if self.recording:
            return

        filename = time.strftime("%Y%m%d_%H%M%S") + ".mp4"
        path = os.path.join(self.video_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            path, fourcc, self.fps, (self.width, self.height)
        )
        self.recording = True
        print(f"[LocalCamera] Start recording: {path}")

    def stop_record(self):
        """停止录像"""
        if not self.recording:
            return

        self.recording = False
        self.video_writer.release()
        self.video_writer = None
        print("[LocalCamera] Stop recording")

    def save_image(self, frame):
        """截图保存"""
        filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
        path = os.path.join(self.image_dir, filename)
        cv2.imwrite(path, frame)
        print(f"[LocalCamera] Image saved: {path}")
