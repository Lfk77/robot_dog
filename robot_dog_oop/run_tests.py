#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotDog Vision System - OOP Refactoring Version
统一测试入口
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <test_name> [--options]")
        print("\nAvailable tests:")
        print("  camera           - 摄像头测试 (本地USB)")
        print("  camera go2       - Go2摄像头测试")
        print("  hand             - 手部检测测试")
        print("  yolo             - YOLO人物检测测试")
        print("  gesture          - 手势控制测试")
        print("  go2_control      - Go2动作控制测试")
        print("  go2_control stub - Go2动作控制(模拟)")
        print("\nOptions:")
        print("  --stub           - 使用存根模式(无机器人时)")
        return

    test_name = sys.argv[1]
    use_stub = "--stub" in sys.argv

    if test_name == "camera":
        from tests.test_camera import TestCamera
        camera_type = sys.argv[2] if len(sys.argv) > 2 else "local"
        tester = TestCamera(camera_type)
        tester.run()

    elif test_name == "hand":
        from tests.test_hand_detector import TestHandDetector
        camera_type = sys.argv[2] if len(sys.argv) > 2 else "local"
        tester = TestHandDetector(camera_type)
        tester.run()

    elif test_name == "yolo":
        from tests.test_yolo_person import TestYoloPerson
        camera_type = sys.argv[2] if len(sys.argv) > 2 else "local"
        tester = TestYoloPerson(camera_type)
        tester.run()

    elif test_name == "gesture":
        from tests.test_gesture_control import TestGestureControl
        camera_type = sys.argv[2] if len(sys.argv) > 2 else "local"
        tester = TestGestureControl(camera_type, use_stub)
        tester.run()

    elif test_name == "go2_control":
        from tests.test_go2_control import TestGo2Control
        tester = TestGo2Control(use_stub)
        tester.run()

    else:
        print(f"Unknown test: {test_name}")


if __name__ == "__main__":
    main()
