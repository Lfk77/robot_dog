# control/base.py
from abc import ABC, abstractmethod


class IActionExecutor(ABC):
    """动作执行器抽象基类"""

    @abstractmethod
    def stand_up(self):
        """起立"""

    @abstractmethod
    def stand_down(self):
        """蹲下"""

    @abstractmethod
    def move(self, vx: float = 0, vy: float = 0, vyaw: float = 0):
        """移动"""

    @abstractmethod
    def stop_move(self):
        """停止移动"""

    @abstractmethod
    def damp(self):
        """阻尼"""

    @abstractmethod
    def recovery(self):
        """恢复姿态"""

    @abstractmethod
    def hello(self):
        """挥手"""

    @abstractmethod
    def release(self):
        """释放资源"""
