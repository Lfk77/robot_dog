# tests/test_yolo_person.py
import cv2
from perception.camera import CameraFactory, CameraType
from perception.detector import YoloPersonDetector


class TestYoloPerson:
    """YOLO人物检测测试"""

    def __init__(self, camera_type: str = "local"):
        self.camera = CameraFactory.create(
            camera_type,
            width=640,
            height=480
        )
        self.detector = YoloPersonDetector()

    def run(self):
        """运行测试"""
        print("===== YOLO Person Detection Test =====")
        print("Press 'q' to quit")

        while True:
            frame = self.camera.read()
            if frame is None:
                break

            detections = self.detector.detect(frame)
            frame = self.detector.draw(frame, detections)

            cv2.putText(
                frame, f"Detections: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("YOLO Person Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        print("[INFO] YOLO person test finished")


if __name__ == "__main__":
    import sys
    camera_type = sys.argv[1] if len(sys.argv) > 1 else "local"
    tester = TestYoloPerson(camera_type)
    tester.run()
