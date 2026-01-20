# utils/buffer.py
from collections import deque
from threading import Lock
from typing import Optional
import numpy as np


class FrameBuffer:
    """帧缓冲区"""

    def __init__(self, max_size: int = 3):
        self._buffer = deque(maxlen=max_size)
        self._lock = Lock()

    def put(self, frame: np.ndarray):
        """添加帧"""
        with self._lock:
            self._buffer.append(frame)

    def get(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        with self._lock:
            if not self._buffer:
                return None
            while len(self._buffer) > 1:
                self._buffer.popleft()
            return self._buffer[-1]

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
