# perception/detector/yolo_gesture.py
from ultralytics import YOLO
import cv2
import time
from typing import List, Dict, Tuple


class YoloGestureDetector:
    """YOLO手势检测器(train9模型)"""

    def __init__(
        self,
        model_path: str = "runs/detect/train9/weights/best.pt",
        device: str = "cpu",
        imgsz: int = 320,
        conf_thres: float = 0.5
    ):
        print(f"[INFO] Loading gesture model: {model_path}")
        self.model = YOLO(model_path).to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf_thres = conf_thres

        self.class_names = {
            0: "hello", 1: "handup", 2: "vectory", 3: "iloveyou",
            4: "fist", 5: "no", 6: "ok", 7: "okay", 8: "callme", 9: "thanks"
        }

        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0)
        ]

        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def detect(self, frame) -> Tuple[List[Dict], List]:
        """检测手势"""
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            device=self.device,
            verbose=False
        )[0]

        detections = []

        if results.boxes is not None:
            for box, cls, conf in zip(
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf
            ):
                class_id = int(cls)
                detections.append({
                    "bbox": box.tolist(),
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": float(conf)
                })

        annotated = self._draw_detections(frame.copy(), detections)
        return detections, annotated

    def _draw_detections(self, frame, detections):
        """绘制检测结果"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cid = det["class_id"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = self.colors[cid % len(self.colors)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - h - 6),
                (x1 + w, y1),
                color,
                -1
            )
            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return frame

    def calc_fps(self) -> float:
        """计算FPS"""
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now
        return self.fps
