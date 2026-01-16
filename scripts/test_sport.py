import time
import cv2
from ultralytics import YOLO  # YOLOv8 官方库

# -----------------------------
# SDK 相关
# -----------------------------
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.common.rpc.client import ClientConfig

# -----------------------------
# 初始化 Go2 控制客户端
# -----------------------------
robot_ip = "192.168.19.222"  # 替换为你的 Go2 IP
sport_client = SportClient(ClientConfig(ip=robot_ip, udp_port=8000))
sport_client.Start()  # 启动运动控制服务
time.sleep(1)
print("[INFO] Connected to Go2")

# -----------------------------
# 手势 -> 动作映射表
# -----------------------------
gesture_to_command = {
    "hello":   "StandUp",      # 张手掌 → 站立
    "fist":   "StandDown",    # 拳头 → 蹲下
    "ok":  "MoveForward",  # 大拇指 → 前进
    "vectory": "MoveBack",     # V 手势 → 后退
    "okay":     "StopMove"      # OK 手势 → 停止
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
# 加载手势识别模型（YOLOv8）
# -----------------------------
model = YOLO("runs/detect/train9/weights/best.pt")  # 替换为你训练好的手势模型权重

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

# -----------------------------
# 主循环
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # YOLOv8 模型预测
        results = model.predict(frame)[0]  # 返回第一张图像结果
        if len(results.boxes) > 0:
            # 取置信度最高的手势
            box = results.boxes[0]
            gesture = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
        else:
            gesture = None
            conf = 0.0

        # 置信度高于 0.8 才处理
        if conf > 0.8:
            if gesture == last_gesture:
                counter += 1
            else:
                last_gesture = gesture
                counter = 1

            # 连续 THRESH 帧识别同一手势才执行动作
            if counter >= THRESH:
                cmd = gesture_to_command.get(gesture)
                if cmd:
                    execute_action(cmd)
                counter = 0

        # 显示识别结果
        text = f"{gesture} ({conf:.2f})" if gesture else "None"
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Go2 Hand Control", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] KeyboardInterrupt received")

finally:
    # 释放摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()

    # 让机器人停止动作，蹲下，关闭控制服务
    sport_client.StopMove()
    sport_client.StandDown()
    sport_client.Stop()
    print("[INFO] Exited safely")
