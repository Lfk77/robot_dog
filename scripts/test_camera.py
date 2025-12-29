#摄像头测试
import cv2
import sys
import os

# === 强制把项目根目录加入 Python 路径 ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from perception.camera import Camera


def main():
    camera = Camera(
        cam_id=0,
        width=640,
        height=480,
        fps=30,
        save_dir="data"
    )

    print("===== Camera Test =====")
    print("Press 's' to save image")
    print("Press 'r' to start/stop recording")
    print("Press 'q' to quit")

    while True:
        frame = camera.read()
        if frame is None:
            print("[ERROR] Cannot read frame from camera")
            break

        # 显示 FPS
        cv2.putText(
            frame,
            f"FPS: {camera.fps_value:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # 显示录制状态
        if camera.recording:
            cv2.putText(
                frame,
                "REC",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            camera.save_image(frame)
        elif key == ord('r'):
            if camera.recording:
                camera.stop_record()
            else:
                camera.start_record()

    camera.release()
    print("[INFO] Camera test finished")


if __name__ == "__main__":
    main()
