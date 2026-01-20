# control/mapper.py
from typing import Dict, Optional


class GestureToActionMapper:
    """手势到动作的映射"""

    DEFAULT_MAPPING = {
        "hello": "stand_up",
        "fist": "stand_down",
        "ok": "move_forward",
        "vectory": "move_back",
        "okay": "stop_move",
        "01_palm": "stand_up",
        "03_fist": "stand_up",
        "07_ok": "move_forward",
        "05_thumb": "move_back",
        "10_down": "stand_up",
    }

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self.mapping = mapping or self.DEFAULT_MAPPING.copy()

    def get_action(self, gesture: str) -> Optional[str]:
        """根据手势获取对应动作"""
        return self.mapping.get(gesture)

    def set_mapping(self, gesture: str, action: str):
        """设置手势映射"""
        self.mapping[gesture] = action

    def load_mapping(self, mapping: Dict[str, str]):
        """加载完整映射配置"""
        self.mapping = mapping

    def get_all_mappings(self) -> Dict[str, str]:
        """获取所有映射"""
        return self.mapping.copy()
