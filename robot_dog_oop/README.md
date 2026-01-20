# RobotDog Vision System - OOP Refactoring

面向对象重构版本，基于原 `robot_dog_vision_system` 项目。

## 目录结构

```
robot_dog_oop/
├── config/              # 配置模块
│   ├── __init__.py
│   └── settings.py      # 配置类
├── perception/          # 感知模块
│   ├── __init__.py
│   ├── camera/          # 摄像头
│   │   ├── __init__.py
│   │   ├── base.py      # 抽象基类
│   │   ├── local_camera.py
│   │   ├── go2_camera.py
│   │   └── factory.py
│   ├── detector/        # 检测器
│   │   ├── __init__.py
│   │   ├── yolo_person.py
│   │   ├── yolo_gesture.py
│   │   └── hand_detector.py
│   └── recognizer/      # 识别器
│       ├── __init__.py
│       ├── gesture_cls.py
│       └── hand_bbox.py
├── control/             # 控制模块
│   ├── __init__.py
│   ├── base.py
│   ├── executor.py
│   ├── stub.py
│   └── mapper.py
├── decision/            # 决策模块
│   ├── __init__.py
│   ├── state_machine.py
│   └── debouncer.py
├── utils/               # 工具模块
│   ├── __init__.py
│   ├── buffer.py
│   ├── fps.py
│   └── image.py
├── tests/               # 测试模块
│   ├── __init__.py
│   ├── test_camera.py
│   ├── test_go2_control.py
│   ├── test_gesture_control.py
│   ├── test_yolo_person.py
│   └── test_hand_detector.py
├── run_tests.py         # 统一入口
└── README.md
```

## 使用方法

### 摄像头测试
```bash
# 本地USB摄像头
python run_tests.py camera

# Go2摄像头
python run_tests.py camera go2
```

### 手部检测测试
```bash
python run_tests.py hand
```

### YOLO人物检测测试
```bash
python run_tests.py yolo
```

### 手势控制测试
```bash
# 本地摄像头
python run_tests.py gesture

# Go2摄像头
python run_tests.py gesture go2

# 存根模式(无机器人)
python run_tests.py gesture --stub
```

### Go2动作控制测试
```bash
python run_tests.py go2_control

# 存根模式
python run_tests.py go2_control --stub
```

## 主要类

### 感知模块

| 类名 | 功能 |
|------|------|
| `LocalCamera` | 本地USB摄像头 |
| `Go2Camera` | Go2 WebRTC摄像头 |
| `CameraFactory` | 摄像头工厂 |
| `YoloPersonDetector` | YOLO人物检测 |
| `YoloGestureDetector` | YOLO手势检测 |
| `MediaPipeHandDetector` | MediaPipe手部检测 |
| `GestureClassifier` | 手势分类器 |

### 控制模块

| 类名 | 功能 |
|------|------|
| `Go2ActionExecutor` | Go2动作执行器 |
| `ActionExecutorStub` | 动作执行存根 |
| `GestureToActionMapper` | 手势-动作映射 |

### 决策模块

| 类名 | 功能 |
|------|------|
| `SystemStateMachine` | 系统状态机 |
| `GestureDebouncer` | 手势防抖器 |

## 配置

在 `config/settings.py` 中修改配置：

```python
# 摄像头配置
CAMERA_LOCAL = {...}
CAMERA_GO2 = {...}

# 模型配置
MODEL_PERSON = {...}
MODEL_GESTURE_DETECT = {...}
MODEL_GESTURE_CLASSIFY = {...}

# SDK配置
SDK = {...}
```

## 注意事项

1. 确保安装依赖：`pip install -r requirements.txt`
2. Go2机器人IP默认为 `192.168.123.161`
3. 使用 `--stub` 参数可在无机器人时测试
