#!/usr/bin/env python3
"""
摄像头实时手部检测（MediaPipe）
- 根据关键点生成 bbox
- bbox 按比例放大
"""

import cv2
import time
import numpy as np
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        camera_id=0,
        width=1280,
        height=720,
        pad_ratio=0.3,   # ⭐ bbox 扩张比例（0.2~0.4 推荐）
        max_hands=1
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.pad_ratio = pad_ratio

        # MediaPipe 初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # FPS 统计
        self.last_time = time.time()
        self.fps = 0

    def calculate_fps(self):
        now = time.time()
        self.fps = 1.0 / (now - self.last_time)
        self.last_time = now
        return self.fps

    def get_hand_bbox(self, landmarks, w, h):
        """
        根据 21 个关键点计算并放大 bbox
        """
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

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("[INFO] 摄像头启动，按 ESC 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 读取摄像头失败")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # 画关键点
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # 计算并放大 bbox
                    x1, y1, x2, y2 = self.get_hand_bbox(hand_landmarks, w, h)

                    # 画 bbox
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Hand",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            fps = self.calculate_fps()
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            cv2.imshow("Hand Detection (Big BBox)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 已退出")


if __name__ == "__main__":
    detector = HandDetector(
        camera_id=0,
        pad_ratio=0.3  # ⭐ 调这里让框更大/更小
    )
    detector.run()
