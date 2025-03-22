import cv2 as cv
import os
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("ERROR!! Camera not opening!!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame!!")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                # Convert normalized coordinates to pixel coordinates
                cx, cy, cz= int(lm.x * w), int(lm.y * h), int(lm.z)
                # Draw circle and text for each landmark
                cv.circle(frame, (cx, cy), 3, (255, 0, 0), cv.FILLED)
                cv.putText(frame, str(cx)+" "+str(cy)+" "+str(cz), (cx + 10, cy + 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv.imshow("Video", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
hands.close()