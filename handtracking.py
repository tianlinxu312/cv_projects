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
        self.lm_list = []
        self.tip_ids = [4, 8, 12, 16, 20]

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

    def get_position(self, img, hand_no=0):
        '''
        :param img: input img
        :param hand_no: Number of hands
        :return:
        '''

        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                # lm is (x, y, z) real-world 3D coordinates in meters with
                # the origin at the hand approximate geometric center.
                # we now try to find the specific pixel position
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def fingers_counter(self, left_hand=False):
        fingers = []
        # check thumb, based on positions on the x-axis
        if left_hand:
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # check all 4 fingers based on positions on the y-axis
        for id in range(1, len(self.tip_ids)):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


