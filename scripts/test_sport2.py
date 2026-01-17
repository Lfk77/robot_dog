#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_sport2.py - 使用手部区域检测+分类方法控制Go2机器人

功能说明：
1. 使用MediaPipe进行手部关键点检测
2. 根据关键点生成手部区域边界框
3. 使用YOLOv8分类模型对手部区域进行手势分类
4. 通过防抖机制控制Go2机器人执行相应动作

与test_sport.py的区别：
- test_sport.py: 使用YOLOv8检测模型直接检测手势
- test_sport2.py: 使用手部区域检测+分类方法（MediaPipe + YOLOv8分类）
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# -----------------------------
# SDK 相关
# -----------------------------
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.common.rpc.client import ClientConfig

# -----------------------------
# 初始化 Go2 控制客户端
# -----------------------------
robot_ip = "192.168.123.161"  # 替换为你的 Go2 IP
sport_client = SportClient(ClientConfig(ip=robot_ip, udp_port=8000))
sport_client.Start()  # 启动运动控制服务
time.sleep(1)
print("[INFO] Connected to Go2")

# -----------------------------
# 手势 -> 动作映射表
# -----------------------------
gesture_to_command = {
    "01_palm":   "StandUp",      # 张手掌 → 站立
    "03_fist":   "StandDown",    # 拳头 → 蹲下
    "07_ok":  "MoveForward",  # 大拇指 → 前进
    "05_thumb": "MoveBack",     # V 手势 → 后退
    "10_down":     "StopMove"      # OK 手势 → 停止
}

# -----------------------------
# 执行动作函数
# -----------------------------
def execute_action(cmd):
    """
    根据动作指令调用 Go2 SDK 执行动作
    """
    if cmd == "StandUp":
        sport_client.StandUp()
    elif cmd == "StandDown":
        sport_client.StandDown()
    elif cmd == "MoveForward":
        sport_client.Move(vx=0.5, vy=0.0, vyaw=0.0)  # 向前移动
    elif cmd == "MoveBack":
        sport_client.Move(vx=-0.5, vy=0.0, vyaw=0.0) # 向后移动
    elif cmd == "StopMove":
        sport_client.StopMove()
    print(f"[ACTION] {cmd}")

# -----------------------------
# 初始化MediaPipe手部检测
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # 只检测一只手
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# 加载手势分类模型（YOLOv8分类模型）
# -----------------------------
print("[INFO] 加载手势分类模型...")
cls_model = YOLO("runs/classify/train3/weights/best.pt")  # 分类模型路径
print("[INFO] 模型加载完成")

# -----------------------------
# 手部区域处理函数
# -----------------------------
def get_hand_bbox(landmarks, w, h, pad_ratio=0.3):
    """根据21个关键点计算bbox并放大"""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)

    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    return x1, y1, x2, y2

def classify_hand(hand_img, imgsz=224):
    """YOLOv8分类模型预测手势"""
    try:
        if hand_img is None or hand_img.size == 0:
            return "unknown", 0.0

        # BGR -> RGB
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        # resize
        hand_img = cv2.resize(hand_img, (imgsz, imgsz))
        # 确保 np.ndarray 且 dtype uint8
        hand_img = np.array(hand_img, dtype=np.uint8)

        # 推理
        results = cls_model(hand_img, device="cpu", verbose=False)[0]

        if hasattr(results, "probs") and results.probs is not None:
            # 获取概率数据 - 注意：results.probs是一个Probs对象，需要访问.data属性
            probs = results.probs.data.cpu().numpy()
            class_id = int(np.argmax(probs))
            conf = float(probs[class_id])
            class_name = results.names[class_id]
            return class_name, conf
        else:
            return "unknown", 0.0

    except Exception as e:
        print(f"[ERROR] 分类模型预测异常: {e}")
        return "error", 0.0

# -----------------------------
# 摄像头初始化
# -----------------------------
cap = cv2.VideoCapture(0)  # 默认摄像头
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# 防抖参数
last_gesture = None  # 上一帧识别的手势
counter = 0          # 连续识别帧计数
THRESH = 5           # 连续帧阈值

# FPS统计
last_time = time.time()
fps = 0

# -----------------------------
# 主循环
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        
        # 计算FPS
        now = time.time()
        fps = 1.0 / (now - last_time)
        last_time = now
        
        # MediaPipe手部检测
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        gesture = None
        conf = 0.0
        bbox = None
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 绘制关键点
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 获取手部边界框
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks, w, h)
                bbox = (x1, y1, x2, y2)
                
                # 裁剪手部区域
                hand_img = frame[y1:y2, x1:x2]
                
                # 分类手势
                gesture, conf = classify_hand(hand_img)
                
                # 绘制边界框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{gesture}: {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                break  # 只处理第一只手

        # 置信度高于0.8才处理
        if conf > 0.8 and gesture:
            if gesture == last_gesture:
                counter += 1
            else:
                last_gesture = gesture
                counter = 1

            # 连续THRESH帧识别同一手势才执行动作
            if counter >= THRESH:
                cmd = gesture_to_command.get(gesture)
                if cmd:
                    execute_action(cmd)
                counter = 0

        # 显示识别结果和FPS
        text = f"{gesture} ({conf:.2f})" if gesture else "None"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Go2 Hand Control (Hand Region + Classification)", frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] KeyboardInterrupt received")

finally:
    # 释放摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()
    
    # 关闭MediaPipe
    hands.close()

    # 让机器人停止动作，蹲下，关闭控制服务
    sport_client.StopMove()
    sport_client.StandDown()
    sport_client.Stop()
    print("[INFO] Exited safely")
