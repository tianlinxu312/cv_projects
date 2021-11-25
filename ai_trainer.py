import pose_estimation as pe
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import utils

# ----------------------
w_cam, h_cam = 1280, 720
# ----------------------


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    detector = pe.PoseTracker()
    count = 0
    # direction up: 0, down: 1 - count 1 if the pose is up + down
    direction = 0

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)
        img = detector.find_pose(img, draw=False)
        # only track hand number 1 and use a larger dot for specific landmark
        landmarks = detector.get_position(img)
        angle_range = [190, 300]

        if landmarks:
            # left arm 11, 13, 15
            left_angle = detector.get_angle(img, 11, 13, 15)
            # right arm 12, 14, 16
            # right_angle = detector.get_angle(img, 12, 14, 16)

            strengh_per = np.interp(left_angle, angle_range, [0, 100])

            if strengh_per == 100:
                if direction == 0:
                    count += 0.5
                    direction = 1
            if strengh_per == 0:
                if direction == 1:
                    count += 0.5
                    direction = 0

            strengh_bar = np.interp(left_angle, angle_range, [500, 150])
            cv2.putText(img, "Energy", (36, 120), cv2.FONT_HERSHEY_COMPLEX, 1, utils.color_picker("lime_green"), 3)
            cv2.rectangle(img, (65, 150), (100, 500), utils.color_picker("lime_green"), 3)
            cv2.rectangle(img, (65, int(strengh_bar)), (100, 500),
                          utils.color_picker("light_green"), cv2.FILLED)
            cv2.putText(img, str(int(strengh_per)) + "%", (48, 550), cv2.FONT_HERSHEY_COMPLEX, 1,
                        utils.color_picker("lime_green"), 3)

            h, w, c = img.shape
            cv2.rectangle(img, (w-200, h-200), (w, h),
                          utils.color_picker("white"), cv2.FILLED)
            cv2.putText(img, str(int(count)), (w-150, h-50), cv2.FONT_HERSHEY_PLAIN, 10,
                        utils.color_picker("blue"), 10)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


