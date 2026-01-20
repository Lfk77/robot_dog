# perception/camera/go2_camera.py
import cv2
import asyncio
import threading
import time
from queue import Queue, Empty
from typing import Optional
import numpy as np

try:
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod
    )
    from aiortc import MediaStreamTrack
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    MediaStreamTrack = None

from .base import ICameraSource


class Go2Camera(ICameraSource):
    """Go2机器人WebRTC摄像头"""

    def __init__(
        self,
        ip: str = "192.168.123.161",
        width: int = 320,
        height: int = 240
    ):
        if not WEBRTC_AVAILABLE:
            print("[WARNING] unitree_webrtc_connect not available")

        self.ip = ip
        self.width = width
        self.height = height

        self.frame_queue = Queue(maxlen=3)
        self.connection = None
        self._is_running = False
        self._fps_value = 0.0
        self._last_fps_time = time.time()
        self._frame_count = 0

        self.async_loop = None
        self.async_thread = None

    def _setup_async_loop(self):
        """设置异步事件循环"""
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True
        )

    def _run_async_loop(self):
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()

    async def _video_callback(self, track):
        """视频帧回调"""
        while self._is_running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")

                if img.shape[1] != self.width or img.shape[0] != self.height:
                    img = cv2.resize(img, (self.width, self.height))

                self.frame_queue.put(img)
                self._update_fps()

            except Exception as e:
                print(f"[Go2Camera] Frame error: {e}")
                break

    def _update_fps(self):
        """更新FPS"""
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self._fps_value = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now

    def connect(self):
        """连接摄像头"""
        if not WEBRTC_AVAILABLE:
            print("[ERROR] WebRTC not available")
            return False

        if self._is_running:
            return True

        self._setup_async_loop()

        async def _connect():
            self.connection = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=self.ip
            )
            await self.connection.connect()
            self.connection.video.switchVideoChannel(True)
            self.connection.video.add_track_callback(self._video_callback)

        try:
            self.async_loop.run_until_complete(_connect())
            self._is_running = True
            self.async_thread.start()
            print(f"[Go2Camera] Connected to {self.ip}")
            return True
        except Exception as e:
            print(f"[Go2Camera] Connection failed: {e}")
            return False

    def read(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=0.1)
        except Empty:
            return None

    def release(self):
        """释放资源"""
        self._is_running = False

        if self.connection:
            async def _disconnect():
                await self.connection.disconnect()

            try:
                self.async_loop.run_until_complete(_disconnect())
            except Exception as e:
                print(f"[Go2Camera] Disconnect error: {e}")

        if self.async_loop:
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)

        cv2.destroyAllWindows()
        print("[Go2Camera] Released")

    def is_opened(self) -> bool:
        return self._is_running and self.connection is not None

    @property
    def fps(self) -> float:
        return self._fps_value

    @property
    def width(self) -> int:
        return self.width

    @property
    def height(self) -> int:
        return self.height
