# perception/camera/base.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ICameraSource(ABC):
    """摄像头抽象基类"""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """读取一帧图像"""

    @abstractmethod
    def release(self):
        """释放资源"""

    @abstractmethod
    def is_opened(self) -> bool:
        """检查是否打开"""

    @property
    @abstractmethod
    def fps(self) -> float:
        """获取FPS"""

    @property
    @abstractmethod
    def width(self) -> int:
        """获取宽度"""

    @property
    @abstractmethod
    def height(self) -> int:
        """获取高度"""
