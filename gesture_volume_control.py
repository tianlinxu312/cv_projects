'''
Can only run on Windows
'''

import handtracking as ht
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()

min_vol = vol_range[0]
max_vol = vol_range[1]
# vol_range = [-65.25, 0]

# ----------------------
w_cam, h_cam = 640, 480
# ----------------------

vol_bar = 400
vol = 0


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    detector = ht.HandTracker()
    ptime = 0

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=True)
        # only track hand number 1 and use a larger dot for specific landmark
        landmarks = detector.get_position(img, hand_no=0)
        if landmarks:
            x1, y1 = landmarks[4][1], landmarks[4][2]
            x2, y2 = landmarks[8][1], landmarks[8][2]
            dx, dy = (x1 + x2) // 2, (y1 + y2) // 2
            color = (255, 0, 255)
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            cv2.circle(img, (dx, dy), 10, color, cv2.FILLED)

            length = math.hypot(x2-x1, y2-y1)
            green = (0, 255, 0)
            blue = (255, 0, 0)
            if length < 50:
                cv2.circle(img, (dx, dy), 10, green, cv2.FILLED)

            # hand range 14 - 320
            hand_range = [15, 335]
            # volume range -65.25 - 0. We interpolates the length to volume range.
            vol = np.interp(length, hand_range, vol_range)
            # set volume to vol
            volume.SetMasterVolumeLevel(vol, None)

            # make a bar
            vol_bar = np.interp(length, hand_range, [400, 150])
            vol_per = np.interp(length, hand_range, [0, 100])
            cv2.rectangle(img, (50, 150), (85, 400), blue, 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), blue, cv2.FILLED)
            cv2.putText(img, str(int(vol_per))+ "%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, blue, 3)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # render time text on the video
        cv2.putText(img, "Frame rates: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


