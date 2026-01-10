from perception.camera import Camera
from perception.hand_detector import HandDetector
from perception.gesture_recognizer import GestureRecognizer
import cv2

def main():
    cam = Camera()
    hand_detector = HandDetector()
    gesture = GestureRecognizer()

    while True:
        frame = cam.read()
        hands = hand_detector.detect(frame)

        for hand in hands:
            g = gesture.recognize(hand)
            x, y = hand["landmarks"][0]
            cv2.putText(frame, g["gesture"], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        frame = hand_detector.draw(frame, hands)
        cv2.imshow("Gesture", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
