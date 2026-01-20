import asyncio
import cv2
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from ultralytics import YOLO

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

# WebRTC 摄像头
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# -----------------------------
# 配置参数 (全局变量)
# -----------------------------
TARGET_WIDTH = 320    # 降低分辨率到320p（原可能是1280）
TARGET_HEIGHT = 240
FRAME_QUEUE_SIZE = 3  # 减小队列大小，减少延迟
INFERENCE_THREADS = 2  # 推理线程数
DISPLAY_SCALE = 2      # 显示时的放大倍数

# -----------------------------
# 初始化 DDS 和控制客户端
# -----------------------------
ChannelFactoryInitialize()
sport_client = SportClient()
time.sleep(1)
print("[INFO] Go2 SportClient initialized")

# -----------------------------
# 订阅状态主题，获取 mode
# -----------------------------
current_state = None
def state_callback(msg: SportModeState_):
    global current_state
    current_state = msg

state_sub = ChannelSubscriber("sportmodestate", SportModeState_)
state_sub.Init(state_callback, 10)

# 等待第一次状态更新
t0 = time.time()
while current_state is None and time.time() - t0 < 3:
    time.sleep(0.1)

# 检查是否卧倒(5)或其他非站立态
if current_state:
    mode = current_state.mode
    print(f"[INFO] Current mode: {mode}")
    if mode == 5:  # LieDown
        print("[INFO] Robot is down → StandUp")
        sport_client.StandUp()
        time.sleep(2)

# -----------------------------
# 手势 -> 动作映射
# -----------------------------
gesture_to_command = {
    "hello": "StandUp",
    "fist": "StandDown",
    "ok": "MoveForward",
    "vectory": "MoveBack",
    "okay": "StopMove"
}

def execute_action(cmd):
    if cmd == "StandUp":
        sport_client.StandUp()
    elif cmd == "StandDown":
        sport_client.StandDown()
    elif cmd == "MoveForward":
        sport_client.Move(vx=0.5)
    elif cmd == "MoveBack":
        sport_client.Move(vx=-0.5)
    elif cmd == "StopMove":
        sport_client.StopMove()
    print(f"[ACTION] {cmd}")

# -----------------------------
# YOLO 手势识别模型 (CPU优化)
# -----------------------------
print("[INFO] Loading YOLO model...")
model = YOLO("runs/detect/train9/weights/best.pt")

# CPU优化设置
model.to('cpu')
# 使用更小的输入尺寸
model_size = 160  # 进一步减小模型输入尺寸

# 推理线程池
inference_executor = ThreadPoolExecutor(max_workers=INFERENCE_THREADS, thread_name_prefix="inference")

# -----------------------------
# WebRTC 摄像头帧队列 (使用deque，更高效)
# -----------------------------
frame_queue = deque(maxlen=FRAME_QUEUE_SIZE)
frame_lock = threading.Lock()
frame_available = threading.Condition()

# FPS统计
fps_stats = deque(maxlen=30)

async def video_callback(track: MediaStreamTrack):
    """WebRTC帧获取回调，降低分辨率"""
    last_frame_time = time.time()
    min_frame_interval = 0.033  # ~30fps的最大帧率
    
    while True:
        try:
            # 计算是否需要跳过帧以维持目标帧率
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            
            if time_since_last < min_frame_interval:
                await asyncio.sleep(min_frame_interval - time_since_last)
                continue
                
            # 获取帧
            frame = await asyncio.wait_for(track.recv(), timeout=0.05)
            
            # 转换为numpy并立即降低分辨率
            img = frame.to_ndarray(format="bgr24")
            
            # 快速降低分辨率
            if img.shape[1] != TARGET_WIDTH or img.shape[0] != TARGET_HEIGHT:
                img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), 
                                interpolation=cv2.INTER_LINEAR)
            
            # 添加到队列
            with frame_lock:
                frame_queue.append(img)
                # 通知有新的帧可用
                with frame_available:
                    frame_available.notify()
            
            last_frame_time = current_time
            await asyncio.sleep(0)  # 让出控制权
            
        except asyncio.TimeoutError:
            continue
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
# 防抖 + 帧率统计
# -----------------------------
last_gesture = None
counter = 0
THRESH = 3  # 降低防抖阈值，减少延迟

# FPS监控器
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

# -----------------------------
# 优化的YOLO推理函数
# -----------------------------
def run_yolo_inference(frame):
    """在独立线程中运行YOLO推理"""
    try:
        # 如果帧已经是小尺寸，可以直接使用
        if frame.shape[1] != model_size:
            inference_frame = cv2.resize(frame, (model_size, model_size))
        else:
            inference_frame = frame.copy()
        
        # 优化推理参数
        results = model.predict(
            inference_frame,
            imgsz=model_size,      # 使用小尺寸
            conf=0.7,              # 置信度阈值
            iou=0.5,
            max_det=1,             # 只检测一个对象
            verbose=False,         # 关闭输出
            device='cpu',
        )[0]
        
        return results
    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        return None

# -----------------------------
# 主显示循环 (多线程优化版)
# -----------------------------
def display_loop_optimized():
    global last_gesture, counter, TARGET_WIDTH, TARGET_HEIGHT
    # 在函数开始处声明所有需要修改的全局变量
    
    # 当前正在处理的帧和推理结果
    current_frame = None
    inference_future = None
    last_inference_time = 0
    inference_interval = 0.1  # 推理间隔，避免过度推理
    
    # 显示相关
    display_frame = None
    gesture_text = "No gesture"
    confidence = 0.0
    
    cv2.namedWindow("Go2 Hand Control", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame_start_time = time.time()
            fps_monitor.update()
            
            # 1. 获取最新帧（非阻塞）
            with frame_lock:
                if frame_queue:
                    current_frame = frame_queue[-1]  # 总是获取最新的帧
                    # 清空旧帧，减少延迟
                    while len(frame_queue) > 1:
                        frame_queue.popleft()
            
            # 如果没有帧可用，等待一小段时间
            if current_frame is None:
                with frame_available:
                    frame_available.wait(timeout=0.01)
                continue
            
            # 2. 异步推理（如果上一轮推理已完成且达到间隔）
            current_time = time.time()
            if (inference_future is None or inference_future.done()) and \
               (current_time - last_inference_time >= inference_interval):
                
                # 提交推理任务
                inference_future = inference_executor.submit(run_yolo_inference, current_frame)
                last_inference_time = current_time
            
            # 3. 检查推理结果（非阻塞）
            if inference_future and inference_future.done():
                try:
                    results = inference_future.result(timeout=0.001)
                    
                    if results and len(results.boxes) > 0:
                        box = results.boxes[0]
                        gesture = results.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        
                        # 防抖逻辑
                        if confidence > 0.7:
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
                            
                            gesture_text = f"{gesture} ({confidence:.2f})"
                        else:
                            gesture_text = "Low confidence"
                            last_gesture = None
                            counter = 0
                    else:
                        gesture_text = "No detection"
                        last_gesture = None
                        counter = 0
                        
                except Exception as e:
                    # print(f"[WARN] Inference result error: {e}")
                    pass
            
            # 4. 准备显示帧（放大显示）
            display_frame = cv2.resize(current_frame, 
                                      (TARGET_WIDTH * DISPLAY_SCALE, TARGET_HEIGHT * DISPLAY_SCALE))
            
            # 5. 添加叠加信息
            fps_text = f"FPS: {fps_monitor.fps:.1f}"
            resolution_text = f"Res: {TARGET_WIDTH}x{TARGET_HEIGHT}"
            
            # FPS显示
            cv2.putText(display_frame, fps_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 分辨率显示
            cv2.putText(display_frame, resolution_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            
            # 手势显示
            cv2.putText(display_frame, gesture_text, (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 推理状态显示
            inference_status = "Inferring..." if inference_future and not inference_future.done() else "Ready"
            cv2.putText(display_frame, inference_status, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            
            # 6. 显示
            cv2.imshow("Go2 Hand Control", display_frame)
            
            # 7. 处理退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):  # 增加分辨率
                TARGET_WIDTH = min(TARGET_WIDTH + 32, 640)
                TARGET_HEIGHT = min(TARGET_HEIGHT + 24, 480)
                print(f"[INFO] Resolution increased to {TARGET_WIDTH}x{TARGET_HEIGHT}")
            elif key == ord('-'):  # 降低分辨率
                TARGET_WIDTH = max(TARGET_WIDTH - 32, 160)
                TARGET_HEIGHT = max(TARGET_HEIGHT - 24, 120)
                print(f"[INFO] Resolution decreased to {TARGET_WIDTH}x{TARGET_HEIGHT}")
            
            # 8. 控制循环频率
            frame_time = time.time() - frame_start_time
            if frame_time < 0.02:  # 如果处理太快，稍微睡眠
                time.sleep(0.02 - frame_time)
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        sport_client.StopMove()
        sport_client.StandDown()
        inference_executor.shutdown(wait=False)
        print("[INFO] Exited safely")

# -----------------------------
# 启动
# -----------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Go2 手势控制 - CPU优化版")
    print("=" * 50)
    print(f"目标分辨率: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"模型输入尺寸: {model_size}x{model_size}")
    print(f"推理线程数: {INFERENCE_THREADS}")
    print(f"队列大小: {FRAME_QUEUE_SIZE}")
    print("操作说明:")
    print("  q - 退出")
    print("  + - 增加分辨率")
    print("  - - 降低分辨率")
    print("=" * 50)
    
    # 启动WebRTC摄像头线程
    loop = asyncio.new_event_loop()
    future = start_camera()
    camera_thread = threading.Thread(target=camera_thread, args=(loop, future), daemon=True)
    camera_thread.start()
    
    # 等待摄像头初始化
    print("[INFO] Waiting for camera initialization...")
    time.sleep(3)
    
    # 启动主显示循环
    try:
        display_loop_optimized()
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
    finally:
        # 清理
        print("[INFO] Cleaning up...")
        loop.call_soon_threadsafe(loop.stop)
        camera_thread.join(timeout=2.0)
        print("[INFO] Shutdown complete")