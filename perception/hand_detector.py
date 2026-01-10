#手部检测
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self,
                 static_mode=False,
                 max_hands=2,
                 detection_conf=0.7,
                 tracking_conf=0.7):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame):
        """输入 BGR 图像，返回手部关键点"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        hands_info = []
        if result.multi_hand_landmarks:
            h, w, _ = frame.shape
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))

                hand_info = {
                    "hand_id": idx,
                    "landmarks": landmarks,
                    "handedness": result.multi_handedness[idx].classification[0].label,
                    "confidence": result.multi_handedness[idx].classification[0].score
                }
                hands_info.append(hand_info)

        return hands_info

    def draw(self, frame, hands_info):
        for hand in hands_info:
            for x, y in hand["landmarks"]:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        return frame
