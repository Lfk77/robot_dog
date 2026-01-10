import cv2
from perception.camera import Camera
from perception.hand_detector import HandDetector
import sys
import os

# === 强制把项目根目录加入 Python 路径 ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
def main():
    cam = Camera()
    detector = HandDetector()

    while True:
        frame = cam.read()
        if frame is None:
            break

        hands = detector.detect(frame)
        frame = detector.draw(frame, hands)

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
