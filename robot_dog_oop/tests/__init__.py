# Tests module
from .test_camera import TestCamera
from .test_go2_control import TestGo2Control
from .test_gesture_control import TestGestureControl
from .test_yolo_person import TestYoloPerson
from .test_hand_detector import TestHandDetector

__all__ = [
    "TestCamera", "TestGo2Control", "TestGestureControl",
    "TestYoloPerson", "TestHandDetector"
]
