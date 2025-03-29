import cv2
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

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    anchor_point = hand_landmarks.landmark[0]

                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z)
                        # Draw circle and text for each landmark
                        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
                        cv2.putText(frame, str(cx - int(anchor_point.x * w)) + " " + str(int(anchor_point.y * h) - cy) + " " + str(id), (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        final = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return final

    def find_position(self, frame, hand_no=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                h, w, c = frame.shape
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
        return landmark_list

