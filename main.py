import cv2 as cv
import os
import mediapipe as mp
import hand_class as hand

hand_dect = hand.Hand()

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR!! Camera not opening!!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame!!")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hand_dect.find_hands(rgb_frame)

    cv.imshow("Video", results)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()