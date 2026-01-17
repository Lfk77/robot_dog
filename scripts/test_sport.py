import time
import cv2
from ultralytics import YOLO

# -----------------------------
# Go2 SDK
# -----------------------------
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# -----------------------------
# 初始化 DDS（⚠️ 必须）
# -----------------------------
print("WARNING: Please ensure there are no obstacles around the robot.")
input("Press Enter to continue...")

ChannelFactoryInitialize(0)   # 和 SDK 示例保持一致

sport_client = SportClient()
sport_client.SetTimeout(10.0)
sport_client.Init()

print("[INFO] Go2 SportClient initialized")

# -----------------------------
# 手势 -> 动作映射
# -----------------------------
gesture_to_command = {
    "hello": "StandUp",
    "fist": "StandDown",
    "ok": "MoveForward",
    "victory": "StopMove",
}

def execute_action(cmd):
    if cmd == "StandUp":
        sport_client.StandUp()
    elif cmd == "StandDown":
        sport_client.StandDown()
    elif cmd == "MoveForward":
        sport_client.Move(0.4, 0.0, 0.0)
    elif cmd == "StopMove":
        sport_client.StopMove()

    print(f"[ACTION] {cmd}")

# -----------------------------
# YOLO 模型
# -----------------------------
model = YOLO("runs/detect/train9/weights/best.pt")

# -----------------------------
# ✅ 使用 Go2 摄像头 RTSP 流
# -----------------------------
GO2_IP = "192.168.123.161"
RTSP_URL = f"rtsp://{GO2_IP}:8554/live"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open Go2 camera RTSP stream")

print("[INFO] Go2 camera stream opened")

# -----------------------------
# 防抖
# -----------------------------
last_gesture = None
counter = 0
THRESH = 5

# -----------------------------
# 主循环
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed")
            continue

        results = model.predict(frame, conf=0.8, verbose=False)[0]

        gesture = None
        conf = 0.0

        if results.boxes:
            box = results.boxes[0]
            gesture = results.names[int(box.cls[0])]
            conf = float(box.conf[0])

        if gesture:
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

        cv2.putText(
            frame,
            f"{gesture} {conf:.2f}" if gesture else "None",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Go2 Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] KeyboardInterrupt")

finally:
    cap.release()
    cv2.destroyAllWindows()
    sport_client.StopMove()
    sport_client.StandDown()
    print("[INFO] Exit safely")
