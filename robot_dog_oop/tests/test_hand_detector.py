# tests/test_hand_detector.py
import cv2
from perception.camera import CameraFactory, CameraType
from perception.detector import MediaPipeHandDetector


class TestHandDetector:
    """手部检测测试"""

    def __init__(self, camera_type: str = "local"):
        self.camera = CameraFactory.create(
            camera_type,
            width=640,
            height=480
        )
        self.detector = MediaPipeHandDetector()

    def run(self):
        """运行测试"""
        print("===== Hand Detector Test =====")
        print("Press 'q' to quit")

        while True:
            frame = self.camera.read()
            if frame is None:
                break

            hands = self.detector.detect(frame)
            frame = self.detector.draw(frame, hands)

            cv2.putText(
                frame, f"Hands: {len(hands)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        self.detector.release()
        cv2.destroyAllWindows()
        print("[INFO] Hand detector test finished")


if __name__ == "__main__":
    import sys
    camera_type = sys.argv[1] if len(sys.argv) > 1 else "local"
    tester = TestHandDetector(camera_type)
    tester.run()
