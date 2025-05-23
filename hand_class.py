import cv2
import mediapipe as md
import numpy as np

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

    def find_hands(self):
        self.results = self.hands.process(self.frame)
        self.mHLM = self.results.multi_hand_landmarks

    def draw_skele(self):
        if self.mHLM:
            for hand_landmarks in self.mHLM:
                self.mp_draw.draw_landmarks(
                    self.frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        final = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        return final

    def show_loc(self):
        if self.mHLM:
            for hand_landmarks in self.results.mHLM:
                anchor_point = hand_landmarks.landmark[0]

                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = self.frame.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z, 4)
                    # Draw circle and text for each landmark
                    # cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
                    cv2.putText(self.frame,
                        str(cx - int(anchor_point.x * w)) + " " + str(int(anchor_point.y * h) - cy) + " " +
                        str(cz) + " " + str(id), (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return self.frame

    def find_dist(self, lm1:int, lm2:int, v ,changeVol=False):
        h, w, c = self.frame.shape
        if self.mHLM:
            for hand_landmarks in self.mHLM:
                x1, y1 = int(hand_landmarks.landmark[lm1].x * w), int(hand_landmarks.landmark[lm1].y * h)
                x2, y2 = int(hand_landmarks.landmark[lm2].x * w), int(hand_landmarks.landmark[lm2].y * h)

                distance = np.hypot(x2-x1, y2 - y1)

                cv2.line(self.frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(self.frame, str(distance), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)

            if changeVol:
                hand_landmark = self.mHLM[0]
                vol_range = v.GetVolumeRange()
                min_vol, max_vol = vol_range[0], vol_range[1]
                vol = np.interp(distance, [20, 400], [-96, 0])
                print(vol)
                v.SetMasterVolumeLevel(vol, None)
        return cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

    def set_frame(self, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def which_hand(self):
        if self.results.multi_handedness:
            for idx, handness in enumerate(self.results.multi_handedness):
                x, y, c = self.frame.shape

                cv2.putText(self.frame, str(handness.classification[0].label), ((x - 100) - (idx * 100), y - 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)