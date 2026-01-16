#!/usr/bin/env python3
"""
测试 train9 手势识别模型（GPU 加速版）
使用摄像头实时检测手势
"""

import cv2
import time
from ultralytics import YOLO
import numpy as np


class Train9Tester:
    def __init__(self,
                 model_path="runs/detect/train9/weights/best.pt",
                 imgsz=320):
        """
        初始化 train9 模型测试器（GPU）

        Args:
            model_path: 模型路径
            imgsz: 推理输入尺寸（推荐 320）
        """
        print(f"[INFO] 加载模型: {model_path}")

        # ⭐ 关键：模型直接放到 GPU
        self.model = YOLO(model_path).to("cuda")
        self.model.fuse()  # Conv + BN 融合，加速推理

        self.imgsz = imgsz

        # 类别名称（必须和 data.yaml 顺序一致）
        self.class_names = {
            0: "hello",
            1: "handup",
            2: "vectory",
            3: "iloveyou",
            4: "fist",
            5: "no",
            6: "ok",
            7: "okay",
            8: "callme",
            9: "thanks"
        }

        # 颜色
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0)
        ]

        # FPS 统计
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    # ==============================
    # 推理
    # ==============================
    def detect(self, frame):
        """
        单帧检测（GPU + FP16）
        """
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=0.5,        # 置信度阈值
            iou=0.5,
            max_det=2,       # 手势一般 1~2 个
            half=True,       # ⭐ FP16
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

        annotated = self.draw_detections(frame.copy(), detections)
        return detections, annotated

    # ==============================
    # 绘制结果
    # ==============================
    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cid = det["class_id"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = self.colors[cid % len(self.colors)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(frame,
                          (x1, y1 - h - 6),
                          (x1 + w, y1),
                          color,
                          -1)

            cv2.putText(frame,
                        label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

        return frame

    # ==============================
    # FPS 计算
    # ==============================
    def calc_fps(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now
        return self.fps

    # ==============================
    # 实时检测
    # ==============================
    def run(self, camera_id=0, width=1280, height=720):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("[INFO] 摄像头启动")
        print("[INFO] 手势类别:", list(self.class_names.values()))
        print("[INFO] 按 ESC 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            detections, vis = self.detect(frame)
            infer_ms = (time.time() - t0) * 1000
            fps = self.calc_fps()

            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Infer: {infer_ms:.1f} ms", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Train9 Gesture Detection (GPU)", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 退出")


# ==============================
# 主函数
# ==============================
def main():
    tester = Train9Tester(
        model_path="runs/detect/train9/weights/best.pt",
        imgsz=320
    )
    tester.run(camera_id=0, width=720, height=360)


if __name__ == "__main__":
    main()
