# control/stub.py
from .base import IActionExecutor


class ActionExecutorStub(IActionExecutor):
    """动作执行器存根(无机器人时模拟)"""

    def stand_up(self):
        print("[Stub] StandUp")

    def stand_down(self):
        print("[Stub] StandDown")

    def move(self, vx: float = 0, vy: float = 0, vyaw: float = 0):
        print(f"[Stub] Move(vx={vx}, vy={vy}, vyaw={vyaw})")

    def stop_move(self):
        print("[Stub] StopMove")

    def damp(self):
        print("[Stub] Damp")

    def recovery(self):
        print("[Stub] RecoveryStand")

    def hello(self):
        print("[Stub] Hello")

    def release(self):
        print("[Stub] Released")
