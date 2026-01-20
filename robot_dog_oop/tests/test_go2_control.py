# tests/test_go2_control.py
from control.executor import Go2ActionExecutor
from control.stub import ActionExecutorStub


class TestGo2Control:
    """Go2控制测试"""

    def __init__(self, use_stub: bool = False):
        if use_stub:
            self.executor = ActionExecutorStub()
        else:
            self.executor = Go2ActionExecutor()
            self.executor.initialize()

        self.actions = {
            "1": ("StandUp", self.executor.stand_up),
            "2": ("StandDown", self.executor.stand_down),
            "3": ("MoveForward", lambda: self.executor.move(vx=0.3)),
            "4": ("MoveBack", lambda: self.executor.move(vx=-0.3)),
            "5": ("MoveLeft", lambda: self.executor.move(vy=0.3)),
            "6": ("MoveRight", lambda: self.executor.move(vy=-0.3)),
            "7": ("RotateLeft", lambda: self.executor.move(vyaw=0.5)),
            "8": ("RotateRight", lambda: self.executor.move(vyaw=-0.5)),
            "9": ("StopMove", self.executor.stop_move),
            "0": ("Hello", self.executor.hello),
        }

    def run(self):
        """运行测试"""
        print("===== Go2 Control Test =====")
        print("Available commands:")
        for key, (name, _) in self.actions.items():
            print(f"  {key}: {name}")
        print("  q: quit")

        try:
            while True:
                cmd = input("Enter command: ").strip().lower()
                if cmd == 'q':
                    break
                if cmd in self.actions:
                    _, action = self.actions[cmd]
                    action()
        finally:
            self.executor.release()
            print("[INFO] Control test finished")


if __name__ == "__main__":
    import sys
    use_stub = "--stub" in sys.argv
    tester = TestGo2Control(use_stub)
    tester.run()
