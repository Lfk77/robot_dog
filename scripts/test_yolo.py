from perception.camera import Camera
from perception.yolo_detector import YoloDetector
import cv2

def main():
    cam = Camera()
    detector = YoloDetector()

    while True:
        frame = cam.read()
        if frame is None:
            break

        dets = detector.detect_person(frame)
        frame = detector.draw_boxes(frame, dets)

        cv2.imshow("YOLO Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
