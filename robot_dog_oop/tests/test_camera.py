# tests/test_camera.py
import cv2
from perception.camera import CameraFactory, CameraType


class TestCamera:
    """摄像头测试"""

    def __init__(self, camera_type: str = "local"):
        self.camera = CameraFactory.create(camera_type)

    def run(self):
        """运行测试"""
        print("===== Camera Test =====")
        print("Press 's' to save image")
        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")

        while True:
            frame = self.camera.read()
            if frame is None:
                print("[ERROR] Cannot read frame")
                break

            cv2.putText(
                frame, f"FPS: {self.camera.fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.imshow("Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.camera.save_image(frame)
            elif key == ord('r'):
                if self.camera.recording:
                    self.camera.stop_record()
                else:
                    self.camera.start_record()

        self.camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera test finished")


if __name__ == "__main__":
    import sys
    camera_type = sys.argv[1] if len(sys.argv) > 1 else "local"
    tester = TestCamera(camera_type)
    tester.run()
