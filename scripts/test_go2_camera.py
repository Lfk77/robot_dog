from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
import cv2
import asyncio
from aiortc import MediaStreamTrack
from queue import Queue
import threading

frame_queue = Queue()

async def video_callback(track: MediaStreamTrack):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        frame_queue.put(img)

async def main():
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")
    await conn.connect()
    conn.video.switchVideoChannel(True)
    conn.video.add_track_callback(video_callback)
    
    while True:
        if not frame_queue.empty():
            img = frame_queue.get()
            cv2.imshow("Go2 Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            await asyncio.sleep(0.01)

    cv2.destroyAllWindows()
    await conn.disconnect()

asyncio.run(main())
