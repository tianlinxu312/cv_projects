import os
import handtracking as ht
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import utils


def main():
    folder_path = "./imgs/header"
    header_list = os.listdir(folder_path)

    overlay_list = []
    for path in header_list:
        if path[-4:] == ".png":
            full_path = os.path.join(folder_path, path)
            img = cv2.imread(full_path)
            overlay_list.append(img)

    header = overlay_list[0]
    draw_color = utils.color_picker("light_pink")
    cap = cv2.VideoCapture(0)

    # ----------------------
    w_cam, h_cam = 1280, 720
    # ----------------------
    cap.set(3, w_cam)
    cap.set(4, h_cam)
    xp, yp = 0, 0

    # create a drawing canvas to keep the drawings
    # it's outside the loop so it doesn't get refreshed after each iter
    drawing_canvas = np.zeros((720, 1280, 3), np.uint8)
    # import hand tracking module
    hand_tracker = ht.HandTracker(detect_con=0.80)
    thickness_per = 15
    thickness_position = (18, 200)
    per_position = (55, 240)
    thick_line_left = (80, 290)
    thick_line_right = (80, 500)

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)

        # put default thickness
        if draw_color == utils.color_picker("black"):
            cv2.putText(img, "Thickness", thickness_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                        utils.color_picker("white"), 3)
            cv2.putText(img, str(thickness_per) + "%", per_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                        utils.color_picker("white"), 3)
            cv2.line(img, thick_line_left, thick_line_right, utils.color_picker("white"), thickness=thickness_per)
        else:
            cv2.putText(img, "Thickness", thickness_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                        draw_color, 3)
            cv2.putText(img, str(thickness_per) + "%", per_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                        draw_color, 3)
            cv2.line(img, thick_line_left, thick_line_right, draw_color, thickness=thickness_per)

        # setting initial header image
        head_h, head_w, head_c = header.shape
        img[:head_h, :head_w, :] = header

        # find hand landmarks
        img = hand_tracker.find_hands(img, draw=False)
        lm_list = hand_tracker.get_position(img)

        if lm_list:
            # find the tip of thumb, index and middle fingers
            thumb1, thumb2 = lm_list[4][1:]
            index1, index2 = lm_list[8][1:]
            middle1, middle2 = lm_list[12][1:]

            # see what fingers up
            fingers = hand_tracker.fingers_counter()

            # use two fingers for selection
            if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
                xp, yp = 0, 0
                # check if tip is in the header
                if index2 < head_h:
                    if index1 < 220:
                        header = overlay_list[1]
                        draw_color = utils.color_picker("light_pink")
                        drawing_canvas = np.zeros((720, 1280, 3), np.uint8)
                    elif 250 < index1 < 450:
                        header = overlay_list[4]
                        draw_color = utils.color_picker("mediumorchid")
                    elif 550 < index1 < 750:
                        header = overlay_list[2]
                        draw_color = utils.color_picker("blue")
                    elif 800 < index1 < 950:
                        header = overlay_list[3]
                        draw_color = utils.color_picker("yellow")
                    elif 1050 < index1 < 1200:
                        header = overlay_list[0]
                        draw_color = utils.color_picker("black")

                if draw_color == utils.color_picker("black"):
                    cv2.circle(img, ((index1 + middle1) // 2, index2), 25, utils.color_picker("white"), 5)
                else:
                    cv2.circle(img, ((index1 + middle1) // 2, index2), 25, draw_color, 5)

            if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                length = math.hypot(index1 - thumb1, index2 - thumb2)
                # length range between index finger and thumb: 15 - 335
                hand_range = [15, 335]
                thickness_per = int(np.interp(length, hand_range, [1, 100]))
                if draw_color == utils.color_picker("black"):
                    cv2.putText(img, "Thickness", thickness_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                                utils.color_picker("white"), 3)
                    cv2.putText(img, str(thickness_per) + "%", per_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                                utils.color_picker("white"), 3)
                    cv2.line(img, thick_line_left, thick_line_right, utils.color_picker("white"), thickness=thickness_per)
                else:
                    cv2.putText(img, "Thickness", thickness_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                                draw_color, 3)
                    cv2.putText(img, str(thickness_per) + "%", per_position, cv2.FONT_HERSHEY_COMPLEX, 1,
                                draw_color, 3)
                    cv2.line(img, thick_line_left, thick_line_right, draw_color, thickness=thickness_per)

            # single index finger for drawing
            if fingers[1] and not fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
                if xp == 0 and yp == 0:
                    xp, yp = index1, index2

                if draw_color == utils.color_picker("black"):
                    cv2.circle(img, (index1, index2), 15, utils.color_picker("white"), cv2.FILLED)
                    cv2.line(img, (xp, yp), (index1, index2), utils.color_picker("white"), thickness=thickness_per)
                else:
                    cv2.circle(img, (index1, index2), 15, draw_color, cv2.FILLED)
                    cv2.line(img, (xp, yp), (index1, index2), draw_color, thickness=thickness_per)
                cv2.line(drawing_canvas, (xp, yp), (index1, index2), draw_color, thickness=thickness_per)

            xp, yp = index1, index2

        img_gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
        _, img_int = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_int, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, drawing_canvas)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()