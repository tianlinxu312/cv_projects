import cv2
import mediapipe as mp


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

    def get_position(self, img, lmk_no=4):
        '''
        :param img: input img
        :param lmk_no: identify a specific landmark by plotting it larger (int: 0-20 or None)
        :return:
        '''

        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
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