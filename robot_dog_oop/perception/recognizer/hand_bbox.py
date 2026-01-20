# perception/recognizer/hand_bbox.py
from typing import Tuple


class HandBBoxExtractor:
    """手部边界框提取器"""

    def __init__(self, pad_ratio: float = 0.3):
        self.pad_ratio = pad_ratio

    def extract(
        self,
        landmarks,
        w: int,
        h: int
    ) -> Tuple[int, int, int, int]:
        """提取手部边界框"""
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        bw = x2 - x1
        bh = y2 - y1

        pad_w = int(bw * self.pad_ratio)
        pad_h = int(bh * self.pad_ratio)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return x1, y1, x2, y2
