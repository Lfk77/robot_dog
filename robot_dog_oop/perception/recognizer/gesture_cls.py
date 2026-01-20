# perception/recognizer/gesture_cls.py
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Optional


class GestureClassifier:
    """手势分类器(train3模型，基于手部区域)"""

    def __init__(
        self,
        model_path: str = "runs/classify/train3/weights/best.pt",
        device: str = "cpu",
        imgsz: int = 224,
        pad_ratio: float = 0.3
    ):
        print(f"[INFO] Loading gesture classifier: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.pad_ratio = pad_ratio

    def classify(self, hand_img: np.ndarray) -> Tuple[str, float]:
        """分类手势"""
        try:
            if hand_img is None or hand_img.size == 0:
                return "unknown", 0.0

            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            hand_img = cv2.resize(hand_img, (self.imgsz, self.imgsz))
            hand_img = np.array(hand_img, dtype=np.uint8)

            results = self.model(hand_img, device=self.device, verbose=False)[0]

            if hasattr(results, "probs") and results.probs is not None:
                probs = results.probs.data.cpu().numpy()
                class_id = int(np.argmax(probs))
                conf = float(probs[class_id])
                class_name = results.names[class_id]
                return class_name, conf
            else:
                return "unknown", 0.0

        except Exception as e:
            print(f"[ERROR] Classification error: {e}")
            return "error", 0.0

    def get_hand_bbox(
        self,
        landmarks,
        w: int,
        h: int
    ) -> Tuple[int, int, int, int]:
        """计算手部边界框"""
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
