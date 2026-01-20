# utils/image.py
import cv2
import numpy as np
from typing import Tuple, Optional


class ImageUtils:
    """图像工具类"""

    @staticmethod
    def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """调整图像大小"""
        return cv2.resize(frame, (width, height))

    @staticmethod
    def crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """裁剪图像"""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]

    @staticmethod
    def draw_bbox(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """绘制边界框"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
        return frame

    @staticmethod
    def draw_text(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        font_scale: float = 0.7,
        thickness: int = 2
    ) -> np.ndarray:
        """绘制文本"""
        cv2.putText(
            frame, text, position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, thickness
        )
        return frame

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """BGR转RGB"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
        """RGB转BGR"""
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def save(frame: np.ndarray, path: str) -> bool:
        """保存图像"""
        return cv2.imwrite(path, frame)
