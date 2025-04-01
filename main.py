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

    hand_dect.find_hands(frame)

    #draws hand skeleton outline
    frame = hand_dect.draw_skele(frame)
    #shows each hand landmark coorinates
    #frame = hand_dect.show_loc(frame)
    #finds distance between thumb and pointer finger
    frame = hand_dect.find_dist(frame, 4, 8)

    cv.imshow("Video", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()