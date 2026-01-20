#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【虚拟机稳定版】
摄像头实时手势识别
- MediaPipe 手部检测
- 关键点生成 bbox（放大）
- YOLOv8 分类模型进行手势分类
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


class GestureRecognizerVM:
    def __init__(
        self,
        cls_model_path,
        camera_id=0,
        width=640,
        height=480,
        pad_ratio=0.3,
        max_hands=1,
        imgsz=224,
        device="cpu"
    ):
        self.width = width
        self.height = height
        self.pad_ratio = pad_ratio
        self.imgsz = imgsz
        self.device = device

        print(f"[INFO] 加载手势分类模型: {cls_model_path}")
        self.model = YOLO(cls_model_path)

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

        self.last_time = time.time()

    def get_fps(self):
        now = time.time()
        fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now
        return int(fps)

    def get_hand_bbox(self, landmarks, w, h):
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        bw, bh = x2 - x1, y2 - y1
        pad_w, pad_h = int(bw * self.pad_ratio), int(bh * self.pad_ratio)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return x1, y1, x2, y2

    def classify(self, hand_img):
        try:
            if hand_img is None or hand_img.size == 0:
                return "none", 0.0

            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            hand_img = cv2.resize(hand_img, (self.imgsz, self.imgsz))
            hand_img = np.ascontiguousarray(hand_img, dtype=np.uint8)

            result = self.model(
                hand_img,
                device=self.device,
                verbose=False
            )[0]

            if result.probs is None:
                return "unknown", 0.0

            probs = result.probs.data.cpu().numpy()
            cls_id = int(np.argmax(probs))
            conf = float(probs[cls_id])
            name = result.names[cls_id]

            return name, conf

        except Exception as e:
            print(f"[WARN] 分类异常: {e}")
            return "error", 0.0

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("[INFO] 摄像头启动，ESC 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 摄像头读取失败")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    self.drawer.draw_landmarks(
                        frame, lm, self.mp_hands.HAND_CONNECTIONS
                    )

                    x1, y1, x2, y2 = self.get_hand_bbox(lm, w, h)
                    hand_img = frame[y1:y2, x1:x2]

                    label, conf = self.classify(hand_img)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            fps = self.get_fps()
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            cv2.imshow("Gesture Recognition (VM)", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 程序结束")


if __name__ == "__main__":
    app = GestureRecognizerVM(
        cls_model_path="runs/classify/train3/weights/best.pt",
        device="cpu",      # ⭐ 虚拟机必须 CPU
        width=640,
        height=480
    )
    app.run()
