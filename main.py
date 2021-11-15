# Hand tracking project - Advanced Computer Vision with Python - Full Course by freeCodeCamp.org
import cv2
import mediapipe as mp
import time
import handtracking as ht


def main():
    cap = cv2.VideoCapture(0)
    detector = ht.HandTracker()
    ptime = 0

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=True)
        # only track hand number 1 and use a larger dot for specific landmark
        landmarks = detector.get_position(img, hand_no=0, lmk_no=4)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # render time text on the video
        cv2.putText(img, "Frame rates: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
