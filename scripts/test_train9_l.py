#!/usr/bin/env python3
"""
测试train9手势识别模型
使用摄像头实时检测手势
"""

import cv2
import time
from ultralytics import YOLO
import numpy as np


class Train9Tester:
    def __init__(self, model_path="runs/train9/weights/best.pt", device="cpu", imgsz=640):
        """
        初始化train9模型测试器
        
        Args:
            model_path: 模型路径
            device: 运行设备 (cuda/cpu)
            imgsz: 推理图像大小，越小越快但精度可能降低
        """
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        
        # 手势类别名称 (从data.yaml中获取)
        self.class_names = {
            0: "hello",
            1: "handup", 
            2: "vectory",
            3: "iloveyou",
            4: "fist",
            5: "no",
            6: "ok",
            7: "okay",
            8: "callme",
            9: "thanks"
        }
        
        # 颜色映射
        self.colors = [
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 0),    # 深蓝
            (0, 128, 0),    # 深绿
            (0, 0, 128),    # 深红
            (128, 128, 0)   # 橄榄色
        ]
        
        self.fps = 30
        self.last_time = time.time()
        self.frame_count = 0
        
    def detect(self, frame):
        """
        检测单帧图像中的手势
        
        Args:
            frame: 输入图像帧
            
        Returns:
            detections: 检测结果列表
            annotated_frame: 标注后的图像帧
        """
        try:
            # 运行推理，使用指定的图像大小
            results = self.model(frame, device=self.device, imgsz=self.imgsz, verbose=False)[0]
            
            detections = []
            
            if results.boxes is not None:
                for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                    class_id = int(cls)
                    confidence = float(conf)
                    
                    detection = {
                        "bbox": box.tolist(),
                        "class_id": class_id,
                        "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                        "confidence": confidence
                    }
                    detections.append(detection)
            
            # 绘制检测结果
            annotated_frame = self.draw_detections(frame.copy(), detections)
            
            return detections, annotated_frame
            
        except Exception as e:
            print(f"检测错误: {e}")
            # 返回空检测结果和原始帧
            return [], frame.copy()
    
    def draw_detections(self, frame, detections):
        """
        在图像上绘制检测框和标签
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            frame: 绘制后的图像
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_id = det["class_id"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # 选择颜色
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                frame, 
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def calculate_fps(self):
        """计算并显示FPS"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        return self.fps
    
    def run_realtime(self, camera_id=0, width=1280, height=720):
        """
        运行实时手势检测
        
        Args:
            camera_id: 摄像头ID
            width: 图像宽度 (默认1280)
            height: 图像高度 (默认720)
        """
        print("初始化摄像头...")
        cap = cv2.VideoCapture(camera_id)
        # 尝试设置更高的分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 获取实际设置的分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {actual_width}x{actual_height}")
        
        # 设置摄像头参数以提高帧率
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 关闭自动对焦
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
        
        print("开始实时手势检测 (按ESC退出)...")
        print("检测的手势类别:", list(self.class_names.values()))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 检测手势并测量时间
                start_time = time.time()
                detections, annotated_frame = self.detect(frame)
                inference_time = time.time() - start_time
                
                # 计算FPS
                fps = self.calculate_fps()
                
                # 显示推理时间
                cv2.putText(
                    annotated_frame,
                    f"Inference: {inference_time*1000:.1f}ms",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 165, 0),  # 橙色
                    2
                )
                
                # 显示FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 显示检测数量
                cv2.putText(
                    annotated_frame,
                    f"Detections: {len(detections)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 显示检测结果
                for i, det in enumerate(detections[:5]):  # 最多显示5个
                    y_pos = 120 + i * 25
                    text = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        1
                    )
                
                # 显示图像
                cv2.imshow("Train9 Gesture Detection", annotated_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    print("用户中断检测")
                    break
                elif key == ord('s'):  # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"gesture_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"保存图像: {filename}")
                elif key == ord(' '):  # 空格键暂停
                    print("暂停，按任意键继续...")
                    cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("检测被中断")
        
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已释放")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试train9手势识别模型")
    parser.add_argument("--model", type=str, default="runs/train9/weights/best.pt",
                       help="模型路径 (默认: runs/train9/weights/best.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="运行设备 cuda/cpu (默认: cpu)")
    parser.add_argument("--camera", type=int, default=0,
                       help="摄像头ID (默认: 0)")
    parser.add_argument("--imgsz", type=int, default=320,
                       help="推理图像大小 (默认: 320，越小越快)")
    parser.add_argument("--width", type=int, default=1280,
                       help="图像宽度 (默认: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="图像高度 (默认: 720)")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = Train9Tester(model_path=args.model, device=args.device, imgsz=args.imgsz)
    
    # 运行实时检测
    tester.run_realtime(
        camera_id=args.camera,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()
