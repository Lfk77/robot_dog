# perception/detector/yolo_person.py
from ultralytics import YOLO
import cv2
from typing import List, Dict


class YoloPersonDetector:
    """YOLO人物检测器"""

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        device: str = "cuda"
    ):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame) -> List[Dict]:
        """检测人物"""
        results = self.model(frame, device=self.device)[0]
        persons = []

        for box, cls, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):
            if int(cls) == 0:
                persons.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class": "person"
                })

        return persons

    def draw(self, frame, detections):
        """绘制检测框"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{det["class"]}:{det["confidence"]:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )
        return frame
