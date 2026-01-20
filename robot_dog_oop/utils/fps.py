# utils/fps.py
from collections import deque
import time


class FpsMonitor:
    """FPS监控器"""

    def __init__(self, window_size: int = 30):
        self._timestamps = deque(maxlen=window_size)
        self._fps = 0.0

    def update(self) -> float:
        """更新FPS"""
        self._timestamps.append(time.time())
        if len(self._timestamps) > 1:
            self._fps = len(self._timestamps) / (
                self._timestamps[-1] - self._timestamps[0]
            )
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self):
        """重置"""
        self._timestamps.clear()
        self._fps = 0.0
