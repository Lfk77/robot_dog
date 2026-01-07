from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, model_path="models/yolov8n.pt", device="cuda"):
        self.model = YOLO(model_path)
        self.device = device

    def detect_person(self, frame):
        results = self.model(frame)[0]
        persons = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0:  # person
                persons.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class": "person"
                })
        return persons

    def draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'{det["class"]}:{det["confidence"]:.2f}',
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return frame
