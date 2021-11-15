# Hand tracking project - Advanced Computer Vision with Python - Full Course by freeCodeCamp.org
import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, max_hs=2, static_img=False, detect_con=0.5, track_con=0.5):
        self.max_hands = max_hs
        self.static_img = static_img
        self.detect_confidence = detect_con
        self.track_confidence = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_img, max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detect_confidence,
                                         min_tracking_confidence=self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        '''
        :param img: Input images
        :param draw: whether to draw the tracking landmarks (Bool)
        :return: return processed img
        '''
        # convert to rgb img because the mp hands module take rgb imgs as inputs
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        # check if there are multiple hands. If so, process them one by one
        if self.results.multi_hand_landmarks:
            for handlmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handlmks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def get_position(self, img, hand_no=0, lmk_no=4):
        '''
        :param img: input img
        :param hand_no: Number of hands
        :param lmk_no: identify a specific landmark by plotting it larger (int: 0-20 or None)
        :return:
        '''

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                # lm is (x, y, z) real-world 3D coordinates in meters with
                # the origin at the hand approximate geometric center.
                # we now try to find the specific pixel position
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                # we can try to identify a specific landmark on image
                if id == lmk_no:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lm_list


