import cv2
from perception.camera import Camera
from perception.yolo_detector import YoloDetector

def main():
    cam = Camera(cam_id=0)
    detector = YoloDetector(model_path="models/yolov8n.pt")

    while True:
        frame = cam.read()
        if frame is None:
            break

        # YOLO检测
        bboxes = detector.detect_person(frame)
        print("检测到人物 bbox:", bboxes)

        # 绘制 bbox
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{box['confidence']:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("YOLO bbox test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
