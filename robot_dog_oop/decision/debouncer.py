# decision/debouncer.py
from typing import Optional, Tuple
import time


class GestureDebouncer:
    """手势防抖器"""

    def __init__(
        self,
        threshold: int = 3,
        cooldown: float = 0.5
    ):
        self.threshold = threshold
        self.cooldown = cooldown

        self._last_gesture = None
        self._counter = 0
        self._last_action_time = 0

    def process(self, gesture: str, confidence: float) -> Tuple[Optional[str], bool]:
        """
        处理手势识别结果

        Returns:
            (action, triggered): (触发的动作, 是否触发)
        """
        current_time = time.time()

        if current_time - self._last_action_time < self.cooldown:
            return None, False

        if confidence < 0.7:
            self._reset()
            return None, False

        if gesture == self._last_gesture:
            self._counter += 1
        else:
            self._last_gesture = gesture
            self._counter = 1

        if self._counter >= self.threshold:
            self._last_action_time = current_time
            action = gesture
            self._reset()
            return action, True

        return None, False

    def reset(self):
        """重置防抖器"""
        self._last_gesture = None
        self._counter = 0

    def _reset(self):
        self._last_gesture = None
        self._counter = 0
