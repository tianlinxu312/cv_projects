import cv2
import mediapipe as mp
import math


class PoseTracker:
    def __init__(self, static_img=False, smooth=True, detect_con=0.5, track_con=0.5):
        self.smooth_lmks = smooth
        self.static_img = static_img
        self.detect_confidence = detect_con
        self.track_confidence = track_con

        self.mp_pose = mp.solutions.pose
        self.poses = self.mp_pose.Pose(static_image_mode=self.static_img,
                                       smooth_landmarks=self.smooth_lmks,
                                       min_detection_confidence=self.detect_confidence,
                                       min_tracking_confidence=self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.lm_list = []

    def find_pose(self, img, draw=True):
        '''
        :param img: Input images
        :param draw: whether to draw the tracking landmarks (Bool)
        :return: return processed img
        '''
        # convert to rgb img because the mp hands module take rgb imgs as inputs
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.poses.process(img_rgb)

        # check if there are multiple hands. If so, process them one by one
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def get_position(self, img):
        '''
        :param img: input img
        :return:
        '''
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # lm is (x, y, z) real-world 3D coordinates in meters with
                # the origin at the hand approximate geometric center.
                # we now try to find the specific pixel position
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def get_angle(self, img, p1, p2, p3, draw=True):
        _, x1, y1 = self.lm_list[p1]
        _, x2, y2 = self.lm_list[p2]
        _, x3, y3 = self.lm_list[p3]

        # compute the angle
        angle = math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2)
        angle = math.degrees(angle)

        if angle < 0:
            angle += 360

        red = [0, 0, 255]
        white = [255, 255, 255]

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 15, red, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, red, cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, red, cv2.FILLED)

            cv2.circle(img, (x1, y1), 13, white, cv2.FILLED)
            cv2.circle(img, (x2, y2), 13, white, cv2.FILLED)
            cv2.circle(img, (x3, y3), 13, white, cv2.FILLED)

            cv2.circle(img, (x1, y1), 8, red, cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, red, cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, red, cv2.FILLED)

        return angle


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0

    pose_tracker = PoseTracker()

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)

        img = pose_tracker.find_pose(img)
        lm_list =pose_tracker.get_position(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()