# tests/test_gesture_control.py
import cv2
from perception.camera import CameraFactory, CameraType
from perception.detector import MediaPipeHandDetector
from perception.recognizer import GestureClassifier
from control.stub import ActionExecutorStub
from control.mapper import GestureToActionMapper
from decision.debouncer import GestureDebouncer
from utils.fps import FpsMonitor


class TestGestureControl:
    """手势控制测试"""

    def __init__(
        self,
        camera_type: str = "local",
        use_stub: bool = False
    ):
        self.camera = CameraFactory.create(
            camera_type,
            width=640,
            height=480
        )

        self.hand_detector = MediaPipeHandDetector()
        self.gesture_classifier = GestureClassifier()
        self.executor = ActionExecutorStub() if use_stub else None
        self.mapper = GestureToActionMapper()
        self.debouncer = GestureDebouncer(threshold=3)
        self.fps_monitor = FpsMonitor()

    def run(self):
        """运行测试"""
        print("===== Gesture Control Test =====")
        print("Press 'q' to quit")

        while True:
            frame = self.camera.read()
            if frame is None:
                break

            hands = self.hand_detector.detect(frame)
            gesture_text = "No gesture"

            for hand in hands:
                h, w, _ = frame.shape
                x1, y1, x2, y2 = self.gesture_classifier.get_hand_bbox(
                    hand["landmarks"][0], w, h
                )
                hand_img = frame[y1:y2, x1:x2]

                gesture, conf = self.gesture_classifier.classify(hand_img)
                action, triggered = self.debouncer.process(gesture, conf)

                if triggered:
                    action_name = self.mapper.get_action(gesture)
                    print(f"[ACTION] Gesture: {gesture} -> Action: {action_name}")

                gesture_text = f"{gesture}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, gesture_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            frame = self.hand_detector.draw(frame, hands)
            fps = self.fps_monitor.update()
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Gesture control test finished")


if __name__ == "__main__":
    import sys
    camera_type = sys.argv[1] if len(sys.argv) > 1 else "local"
    use_stub = "--stub" in sys.argv

    tester = TestGestureControl(camera_type, use_stub)
    tester.run()
