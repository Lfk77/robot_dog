# control/executor.py
import time
from typing import Optional

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

from .base import IActionExecutor


class Go2ActionExecutor(IActionExecutor):
    """Go2动作执行器"""

    ACTIONS = {
        "damp": 0,
        "stand_up": 1,
        "stand_down": 2,
        "move_forward": 3,
        "move_lateral": 4,
        "move_rotate": 5,
        "stop_move": 6,
        "hand_stand": 7,
        "balance_stand": 9,
        "recovery": 10,
        "left_flip": 11,
        "back_flip": 12,
        "free_walk": 13,
        "free_bound": 14,
        "free_avoid": 15,
        "walk_upright": 17,
        "cross_step": 18,
        "free_jump": 19
    }

    def __init__(
        self,
        network_interface: Optional[str] = None,
        timeout: float = 10.0
    ):
        if not SDK_AVAILABLE:
            print("[WARNING] unitree_sdk2py not available")

        self.network_interface = network_interface
        self.timeout = timeout
        self._initialized = False
        self.client = None

    def initialize(self):
        """初始化SDK"""
        if not SDK_AVAILABLE:
            print("[ERROR] SDK not available")
            return

        if self._initialized:
            return

        if self.network_interface:
            ChannelFactoryInitialize(0, self.network_interface)
        else:
            ChannelFactoryInitialize(0)

        self.client = SportClient()
        self.client.SetTimeout(self.timeout)
        self.client.Init()
        self._initialized = True
        print("[Go2ActionExecutor] Initialized")

    def _check_init(self):
        if not self._initialized:
            self.initialize()

    def stand_up(self):
        self._check_init()
        self.client.StandUp()

    def stand_down(self):
        self._check_init()
        self.client.StandDown()

    def move(self, vx: float = 0, vy: float = 0, vyaw: float = 0):
        self._check_init()
        self.client.Move(vx, vy, vyaw)

    def stop_move(self):
        self._check_init()
        self.client.StopMove()

    def damp(self):
        self._check_init()
        self.client.Damp()

    def recovery(self):
        self._check_init()
        self.client.RecoveryStand()

    def hello(self):
        self._check_init()
        self.client.Hello()

    def hand_stand(self, enable: bool = True):
        self._check_init()
        self.client.HandStand(enable)

    def balance_stand(self):
        self._check_init()
        self.client.BalanceStand()

    def left_flip(self):
        self._check_init()
        self.client.LeftFlip()

    def back_flip(self):
        self._check_init()
        self.client.BackFlip()

    def free_walk(self, enable: bool = True):
        self._check_init()
        self.client.FreeWalk(enable)

    def free_bound(self, enable: bool = True):
        self._check_init()
        self.client.FreeBound(enable)

    def free_avoid(self, enable: bool = True):
        self._check_init()
        self.client.FreeAvoid(enable)

    def walk_upright(self, enable: bool = True):
        self._check_init()
        self.client.WalkUpright(enable)

    def cross_step(self, enable: bool = True):
        self._check_init()
        self.client.CrossStep(enable)

    def free_jump(self, enable: bool = True):
        self._check_init()
        self.client.FreeJump(enable)

    def release(self):
        """释放资源"""
        if self._initialized:
            try:
                self.stop_move()
                self.stand_down()
            except Exception:
                pass
            self._initialized = False
            print("[Go2ActionExecutor] Released")
