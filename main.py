import cv2 as cv
import hand_class as hand
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

hand_dect = hand.Hand()

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("ERROR!! Camera not opening!!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame!!")
        break

    hand_dect.set_frame(frame)

    hand_dect.find_hands()

    #draws hand skeleton outline
    skele_frame = hand_dect.draw_skele()
    #shows each hand landmark coorinates
    #loc_frame = hand_dect.show_loc()
    #finds distance between thumb and pointer finger
    dist_frame = hand_dect.find_dist(4, 8, volume,False)

    #dist_frame = hand_dect.find_dist(16, 20, volume, False)

    dist_frame = hand_dect.which_hand()

    cv.imshow("Video", dist_frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()