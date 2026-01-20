# decision/state_machine.py
from enum import Enum, auto
from typing import Callable, Optional


class SystemState(Enum):
    """系统状态"""
    IDLE = auto()
    DETECTING = auto()
    TRACKING = auto()
    EXECUTING = auto()
    ERROR = auto()


class SystemStateMachine:
    """系统状态机"""

    def __init__(self):
        self._state = SystemState.IDLE
        self._callbacks = {}

    @property
    def state(self) -> SystemState:
        return self._state

    def transition(self, new_state: SystemState):
        """状态转换"""
        old_state = self._state
        self._state = new_state
        print(f"[StateMachine] {old_state.name} -> {new_state.name}")

        if new_state in self._callbacks:
            self._callbacks[new_state]()

    def on_enter(self, state: SystemState, callback: Callable):
        """注册状态进入回调"""
        self._callbacks[state] = callback

    def reset(self):
        """重置状态"""
        self.transition(SystemState.IDLE)
