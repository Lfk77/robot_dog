#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_sport_ethernet_webrtc.py
- Go2 以太网直连动作控制（指定网卡名）
- WebRTC 摄像头 + MediaPipe 手部检测 + YOLO 分类手势
- 显示手部 bbox 和手势标签 + FPS
"""

import asyncio
import cv2
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# -----------------------------
# Unitree SDK
# -----------------------------
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# WebRTC 摄像头
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# -----------------------------
# 配置参数
# -----------------------------
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
FRAME_QUEUE_SIZE = 3
INFERENCE_THREADS = 2
DISPLAY_SCALE = 2
MODEL_SIZE = 160
THRESH = 3
ETH_INTERFACE = "enx6c1ff70a9027"  # 以太网网卡名

# -----------------------------
# 初始化 Go2（以太网直连）
# -----------------------------
ChannelFactoryInitialize(0, ETH_INTERFACE)
sport_client = SportClient()
sport_client.Init()  # 官方 SDK 要求 Init()
time.sleep(1)
print("[INFO] Go2 SportClient initialized via Ethernet")

# -----------------------------
# 订阅状态（卧倒自动站起）
# -----------------------------
current_state = None
def state_callback(msg: SportModeState_):
    global current_state
    current_state = msg

state_sub = ChannelSubscriber("sportmodestate", SportModeState_)
state_sub.Init(state_callback, 10)

t0 = time.time()
while current_state is None and time.time() - t0 < 3:
    time.sleep(0.1)

if current_state and current_state.mode == 5:  # LieDown
    print("[INFO] Robot is down → StandUp")
    sport_client.StandUp()
    time.sleep(2)

# -----------------------------
# 手势 -> 动作映射
# -----------------------------
gesture_to_command = {
    "01_palm":   "StandUp",
    "03_fist":   "StandUp",
    "07_ok":     "MoveForward",
    "05_thumb":  "MoveBack",
    "10_down":   "StandUp",
}

def execute_action(cmd):
    try:
        if cmd == "StandUp":
            sport_client.Hello()
        elif cmd == "StandDown":
            sport_client.StandDown()
        elif cmd == "MoveForward":
            sport_client.Move(vx=0.5, vy=0.0, vyaw=0.0)
        elif cmd == "MoveBack":
            sport_client.Move(vx=-0.5, vy=0.0, vyaw=0.0)
        elif cmd == "StopMove":
            sport_client.StopMove()
        print(f"[ACTION] {cmd}")
    except Exception as e:
        print(f"[ERROR] Action failed ({cmd}): {e}")

# -----------------------------
# MediaPipe 初始化
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# YOLO 手势分类模型
# -----------------------------
print("[INFO] Loading YOLO hand classification model...")
cls_model = YOLO("runs/classify/train3/weights/best.pt")
cls_model.to('cpu')
print("[INFO] Model loaded")

# -----------------------------
# WebRTC 摄像头队列
# -----------------------------
frame_queue = deque(maxlen=FRAME_QUEUE_SIZE)
frame_lock = threading.Lock()
frame_available = threading.Condition()

# FPS 监控
class FPSMonitor:
    def __init__(self, window_size=10):
        self.timestamps = deque(maxlen=window_size)
        self.fps = 0
    def update(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > 1:
            self.fps = len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
        return self.fps
fps_monitor = FPSMonitor()

async def video_callback(track: MediaStreamTrack):
    while True:
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
            with frame_lock:
                frame_queue.append(img)
                with frame_available:
                    frame_available.notify()
            await asyncio.sleep(0)
        except Exception as e:
            print(f"[ERROR] Frame capture error: {e}")
            break

async def start_camera():
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")
    await conn.connect()
    conn.video.switchVideoChannel(True)
    conn.video.add_track_callback(video_callback)
    return conn

def camera_thread(loop, future):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(future)
    loop.run_forever()

# -----------------------------
# 手部 bbox / 分类
# -----------------------------
inference_executor = ThreadPoolExecutor(max_workers=INFERENCE_THREADS)

def get_hand_bbox(landmarks, w, h, pad_ratio=0.3):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1, y1 = int(min(xs)*w), int(min(ys)*h)
    x2, y2 = int(max(xs)*w), int(max(ys)*h)
    bw, bh = x2-x1, y2-y1
    pad_w, pad_h = int(bw*pad_ratio), int(bh*pad_ratio)
    x1, y1 = max(0,x1-pad_w), max(0,y1-pad_h)
    x2, y2 = min(w,x2+pad_w), min(h,y2+pad_h)
    return x1, y1, x2, y2

def classify_hand(hand_img):
    try:
        if hand_img is None or hand_img.size == 0:
            return "unknown", 0.0
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        hand_img = cv2.resize(hand_img, (MODEL_SIZE, MODEL_SIZE))
        results = cls_model(hand_img, device='cpu', verbose=False)[0]
        if hasattr(results, "probs") and results.probs is not None:
            probs = results.probs.data.cpu().numpy()
            cid = int(np.argmax(probs))
            conf = float(probs[cid])
            return results.names[cid], conf
        else:
            return "unknown", 0.0
    except Exception as e:
        print(f"[ERROR] Classification error: {e}")
        return "error", 0.0

# -----------------------------
# 主显示循环
# -----------------------------
last_gesture = None
counter = 0

def display_loop():
    global last_gesture, counter
    current_frame = None
    inference_future = None
    last_inf_time = 0
    inf_interval = 0.1

    cv2.namedWindow("Go2 Hand Control", cv2.WINDOW_NORMAL)

    while True:
        with frame_lock:
            if frame_queue:
                current_frame = frame_queue[-1]
                while len(frame_queue) > 1:
                    frame_queue.popleft()
        if current_frame is None:
            with frame_available:
                frame_available.wait(timeout=0.01)
            continue

        h, w, _ = current_frame.shape

        # MediaPipe 手部检测
        rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_text = "No gesture"
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            x1, y1, x2, y2 = get_hand_bbox(hand_landmarks, w, h)
            hand_img = current_frame[y1:y2, x1:x2]

            # 异步 YOLO 分类
            if (inference_future is None or inference_future.done()) and (time.time() - last_inf_time >= inf_interval):
                inference_future = inference_executor.submit(classify_hand, hand_img)
                last_inf_time = time.time()
            
            if inference_future and inference_future.done():
                gesture, conf = inference_future.result()
                if conf > 0.7:
                    if gesture == last_gesture:
                        counter += 1
                    else:
                        last_gesture = gesture
                        counter = 1
                    if counter >= THRESH:
                        cmd = gesture_to_command.get(gesture)
                        if cmd:
                            execute_action(cmd)
                        counter = 0
                    gesture_text = f"{gesture} ({conf:.2f})"
                else:
                    gesture_text = "Low confidence"
                    last_gesture = None
                    counter = 0

            # 绘制 bbox 和手部关键点
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            mp_draw.draw_landmarks(current_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # FPS 显示
        fps_monitor.update()
        cv2.putText(current_frame, f"FPS: {fps_monitor.fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.putText(current_frame, gesture_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        display_frame = cv2.resize(current_frame, (TARGET_WIDTH*DISPLAY_SCALE, TARGET_HEIGHT*DISPLAY_SCALE))
        cv2.imshow("Go2 Hand Control", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 清理
    cv2.destroyAllWindows()
    sport_client.StopMove()
    sport_client.StandDown()
    inference_executor.shutdown(wait=False)
    hands.close()
    print("[INFO] Exited safely")

# -----------------------------
# 启动
# -----------------------------
if __name__ == "__main__":
    print("[INFO] Starting hand control (Ethernet + WebRTC)...")
    loop = asyncio.new_event_loop()
    future = start_camera()
    cam_thread = threading.Thread(target=camera_thread, args=(loop, future), daemon=True)
    cam_thread.start()
    time.sleep(3)  # 等待摄像头初始化
    display_loop()
    loop.call_soon_threadsafe(loop.stop)
    cam_thread.join(timeout=2.0)
