# config/settings.py
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Settings:
    """项目配置"""

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RUNS_DIR = PROJECT_ROOT / "runs"

    CAMERA_LOCAL = {
        "cam_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    }

    CAMERA_GO2 = {
        "ip": "192.168.123.161",
        "width": 320,
        "height": 240
    }

    MODEL_PERSON = {
        "path": "models/yolov8n.pt",
        "device": "cuda"
    }

    MODEL_GESTURE_DETECT = {
        "path": "runs/detect/train9/weights/best.pt",
        "device": "cpu",
        "imgsz": 320,
        "conf_thres": 0.5
    }

    MODEL_GESTURE_CLASSIFY = {
        "path": "runs/classify/train3/weights/best.pt",
        "device": "cpu",
        "imgsz": 224,
        "pad_ratio": 0.3
    }

    SDK = {
        "network_interface": None,
        "timeout": 10.0
    }

    DEBOUNCER = {
        "threshold": 3,
        "cooldown": 0.5
    }

    HAND_DETECTOR = {
        "max_hands": 2,
        "detection_conf": 0.7,
        "tracking_conf": 0.7
    }

    ACTION_MAPPING = {
        "hello": "stand_up",
        "fist": "stand_down",
        "ok": "move_forward",
        "vectory": "move_back",
        "okay": "stop_move",
    }

    @classmethod
    def get_camera_local(cls) -> Dict[str, Any]:
        return cls.CAMERA_LOCAL.copy()

    @classmethod
    def get_camera_go2(cls) -> Dict[str, Any]:
        return cls.CAMERA_GO2.copy()

    @classmethod
    def get_model_person(cls) -> Dict[str, Any]:
        return cls.MODEL_PERSON.copy()

    @classmethod
    def get_model_gesture_detect(cls) -> Dict[str, Any]:
        return cls.MODEL_GESTURE_DETECT.copy()

    @classmethod
    def get_model_gesture_classify(cls) -> Dict[str, Any]:
        return cls.MODEL_GESTURE_CLASSIFY.copy()

    @classmethod
    def get_action_mapping(cls) -> Dict[str, str]:
        return cls.ACTION_MAPPING.copy()

    @classmethod
    def load_yaml(cls, config_path: str) -> Dict:
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
