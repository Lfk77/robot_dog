#!/usr/bin/env python3
"""
测试train9手势识别模型
使用摄像头实时检测手势 - 离线版本
"""

import os
# 在导入任何包之前设置环境变量
os.environ['YOLO_NO_DOWNLOADS'] = '1'
os.environ['GITHUB_ASSETS'] = ''
os.environ['ULTRALYTICS_NO_DOWNLOADS'] = '1'
os.environ['GITHUB_TOKEN'] = ''

import cv2
import time
import numpy as np

# 尝试直接加载模型而不触发在线下载
def load_model_offline(model_path):
    """
    离线加载YOLO模型的替代方法
    """
    print("使用离线模式加载模型...")
    
    try:
        # 方法1：直接使用PyTorch加载
        import torch
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 根据YOLO模型结构创建模型
        from ultralytics.nn.tasks import DetectionModel
        
        # 获取模型配置
        model_config = checkpoint.get('model', checkpoint.get('yaml', None))
        
        if model_config is None:
            # 尝试从checkpoint中推断模型结构
            model = DetectionModel(cfg='yolov8n.yaml')  # 使用默认配置
        else:
            model = DetectionModel(cfg=model_config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        model.eval()
        
        print("模型离线加载成功！")
        return model
        
    except Exception as e:
        print(f"离线加载失败: {e}")
        raise

class Train9Tester:
    def __init__(self, model_path="runs/detect/train9/weights/best.pt", device="cpu", imgsz=640):
        """
        初始化train9模型测试器
        """
        print(f"加载模型: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 先尝试正常加载（在已经设置环境变量的情况下）
            self.model = self._load_yolo_safely(model_path)
            print("模型加载成功！")
            
        except Exception as e:
            print(f"YOLO加载失败: {e}")
            print("尝试备用加载方式...")
            self.model = load_model_offline(model_path)
            self._use_custom_predict = True
            print("使用备用加载方式成功！")
        else:
            self._use_custom_predict = False
        
        self.device = device
        self.imgsz = imgsz
        
        # 手势类别名称
        self.class_names = {
            0: "hello", 1: "handup", 2: "vectory", 3: "iloveyou",
            4: "fist", 5: "no", 6: "ok", 7: "okay", 8: "callme", 9: "thanks"
        }
        
        # 颜色映射
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0)
        ]
        
        self.fps = 30
        self.last_time = time.time()
        self.frame_count = 0
    
    def _load_yolo_safely(self, model_path):
        """
        安全加载YOLO模型，避免在线下载
        """
        # 临时禁用requests以防止在线请求
        import requests
        from functools import wraps
        
        # 保存原始函数
        original_get = requests.get
        original_head = requests.head
        
        def disabled_get(*args, **kwargs):
            """禁用的requests.get函数"""
            url = args[0] if args else kwargs.get('url', '')
            if 'github.com' in str(url) or 'ultralytics.com' in str(url):
                # 对于模型文件请求，返回本地文件
                if str(url).endswith('.pt'):
                    class MockResponse:
                        def __init__(self):
                            self.status_code = 200
                        def raise_for_status(self):
                            pass
                    return MockResponse()
                raise ConnectionError("Online download is disabled. Using local model only.")
            return original_get(*args, **kwargs)
        
        def disabled_head(*args, **kwargs):
            """禁用的requests.head函数"""
            url = args[0] if args else kwargs.get('url', '')
            if 'github.com' in str(url) or 'ultralytics.com' in str(url):
                class MockResponse:
                    def __init__(self):
                        self.status_code = 200
                    def raise_for_status(self):
                        pass
                return MockResponse()
            return original_head(*args, **kwargs)
        
        # 替换requests函数
        requests.get = disabled_get
        requests.head = disabled_head
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path, verbose=False)
            return model
        finally:
            # 恢复原始函数
            requests.get = original_get
            requests.head = original_head
    
    def detect(self, frame):
        """
        检测单帧图像中的手势
        """
        try:
            if self._use_custom_predict:
                # 使用自定义预测
                detections = self._custom_detect(frame)
            else:
                # 使用YOLO的predict方法
                results = self.model(frame, device=self.device, imgsz=self.imgsz, verbose=False)[0]
                detections = self._parse_results(results)
            
            # 绘制检测结果
            annotated_frame = self.draw_detections(frame.copy(), detections)
            
            return detections, annotated_frame
            
        except Exception as e:
            print(f"检测错误: {e}")
            return [], frame.copy()
    
    def _parse_results(self, results):
        """
        解析YOLO结果
        """
        detections = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
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
        
        return detections
    
    def _custom_detect(self, frame):
        """
        自定义检测函数（用于备用模型）
        """
        # 这里实现自定义检测逻辑
        # 由于模型结构不同，这里需要根据实际模型调整
        # 这是一个简单示例，实际使用时需要根据你的模型实现
        
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 转换图像格式
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            if self.device == 'cuda':
                img_tensor = img_tensor.cuda()
                self.model = self.model.cuda()
            
            outputs = self.model(img_tensor)
        
        # 这里需要根据实际模型输出解析检测结果
        # 这是一个示例，实际需要根据模型调整
        detections = []
        
        # 示例：假设输出是 [batch, num_boxes, 6] 格式
        # 其中每个box是 [x1, y1, x2, y2, conf, class]
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 3:
            for box in outputs[0]:
                if box[4] > 0.25:  # 置信度阈值
                    detection = {
                        "bbox": box[:4].tolist(),
                        "class_id": int(box[5]),
                        "class_name": self.class_names.get(int(box[5]), f"class_{int(box[5])}"),
                        "confidence": float(box[4])
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        在图像上绘制检测框和标签
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
        """
        print("初始化摄像头...")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {actual_width}x{actual_height}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("开始实时手势检测 (按ESC退出)...")
        print("检测的手势类别:", list(self.class_names.values()))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 检测手势
                start_time = time.time()
                detections, annotated_frame = self.detect(frame)
                inference_time = time.time() - start_time
                
                # 计算FPS
                fps = self.calculate_fps()
                
                # 显示信息
                cv2.putText(
                    annotated_frame,
                    f"Inference: {inference_time*1000:.1f}ms",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 165, 0),
                    2
                )
                
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
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
                for i, det in enumerate(detections[:5]):
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
                cv2.imshow("Train9 Gesture Detection - OFFLINE MODE", annotated_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    print("用户中断检测")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"gesture_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"保存图像: {filename}")
                elif key == ord(' '):
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
    
    parser = argparse.ArgumentParser(description="测试train9手势识别模型（离线版）")
    parser.add_argument("--model", type=str, default="runs/detect/train9/weights/best.pt",
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