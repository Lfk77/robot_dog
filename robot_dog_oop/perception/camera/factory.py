# perception/camera/factory.py
from enum import Enum
from typing import Union
from .base import ICameraSource
from .local_camera import LocalCamera
from .go2_camera import Go2Camera


class CameraType(Enum):
    LOCAL = "local"
    GO2 = "go2"


class CameraFactory:
    """摄像头工厂"""

    @staticmethod
    def create(
        camera_type: Union[CameraType, str],
        **kwargs
    ) -> ICameraSource:
        """创建摄像头实例"""
        if isinstance(camera_type, str):
            try:
                camera_type = CameraType(camera_type)
            except ValueError:
                raise ValueError(f"Unknown camera type: {camera_type}")

        if camera_type == CameraType.LOCAL:
            return LocalCamera(**kwargs)
        elif camera_type == CameraType.GO2:
            return Go2Camera(**kwargs)
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
