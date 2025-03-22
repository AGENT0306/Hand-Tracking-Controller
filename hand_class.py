import mediapipe as md

class Hand:
    def __init__(self):
        self.mp_hands = md.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = md.solutions.drawing_utils
