#手势识别
import math

class GestureRecognizer:
    def __init__(self):
        pass

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def recognize(self, hand_info):
        lm = hand_info["landmarks"]

        # 简单示例：判断“张开手”
        palm = lm[0]
        fingertips = [lm[4], lm[8], lm[12], lm[16], lm[20]]

        dists = [self.distance(palm, tip) for tip in fingertips]

        if min(dists) > 80:
            return {"gesture": "OPEN_PALM", "confidence": 0.8}

        return {"gesture": "UNKNOWN", "confidence": 0.2}
