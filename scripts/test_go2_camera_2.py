import cv2
import asyncio
import logging
import threading
from queue import Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# 日志等级设为 WARNING，减少控制台输出
logging.basicConfig(level=logging.WARNING)

def main():
    frame_queue = Queue()

    # 创建 WebRTC 连接（直连以太网 STA）
    conn = UnitreeWebRTCConnection(connectionMethod=WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")

    # 异步回调，用于接收视频帧 
    async def recv_video(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    # 在单独线程中运行 asyncio loop
    def run_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                print("[INFO] 连接 Go2 摄像头...")
                await conn.connect()
                print("[INFO] 连接成功！")

                # 打开视频通道
                conn.video.switchVideoChannel(True)

                # 添加视频回调
                conn.video.add_track_callback(recv_video)

            except Exception as e:
                print(f"[ERROR] WebRTC 连接异常: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    t.start()

    # OpenCV 窗口显示视频
    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                cv2.imshow("Go2 Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 没有帧就稍微休眠，降低 CPU 占用
                asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("[INFO] 用户中断")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] 关闭连接...")
        loop.call_soon_threadsafe(loop.stop)
        t.join()

if __name__ == "__main__":
    main()
