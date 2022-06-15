from __future__ import print_function
from math import hypot
import cv2
import time
import mediapipe as mp
import numpy as np
import keyboard
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist

    def distancebetweenindexandthumb(self, img, lmlist=[]):
        thumb_x = lmlist[4][1]
        thumb_y = lmlist[4][2]
        index_x = lmlist[8][1]
        index_y = lmlist[8][2]

        x_coordinate_diff = index_x - thumb_x
        y_coordinate_diff = index_y - thumb_y
        fingerdistance = hypot(x_coordinate_diff, y_coordinate_diff)

        cv2.circle(img, (thumb_x, thumb_y), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
        cv2.circle(img, (index_x, index_y), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
        cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)

        volbar = np.interp(fingerdistance, [30, 350], [400, 150])
        volper = np.interp(fingerdistance, [30, 350], [0, 100])

        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume_master = np.interp(volper, [0, 100], [-60, 0])
        volume.SetMasterVolumeLevel(volume_master, None)


def volumecontrol():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    pTime = 0
    while True:
        success, image = cap.read()
        cTIme = time.time()
        fps = 1 / (cTIme - pTime)
        pTime = cTIme
        cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image = tracker.handsFinder(image, draw=False)
        lmList = tracker.positionFinder(image, draw=False)
        if len(lmList) != 0:
            tracker.distancebetweenindexandthumb(image, lmList)
        cv2.imshow("Video", image)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    volumecontrol()
