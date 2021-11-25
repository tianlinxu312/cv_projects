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

    def fingers_counter(self, left_hand=True):
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


# def main():
#     cap = cv2.VideoCapture(0)
#     ptime = 0
#
#     pose_tracker = PoseTracker()
#
#     while True:
#         success, img = cap.read()
#         # take the mirrored img if needed
#         img = cv2.flip(img, 1)
#
#         img = pose_tracker.find_pose(img)
#         lm_list =pose_tracker.get_position(img)
#
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
#
#
# if __name__ == '__main__':
#     main()

class FaceMesh:
    def __init__(self, static=False, max_faces=2, detect_con=0.5, track_con=0.5, drawing_thickness=1, cir_radius=2):
        self.static_img = static
        self.max_faces = max_faces
        self.detect_confidence = detect_con
        self.drawing_thickness = drawing_thickness
        self.circle_radius = cir_radius
        self.track_confidence = track_con

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_img, max_num_faces=self.max_faces,
                                                    min_detection_confidence=self.detect_confidence,
                                                    min_tracking_confidence=self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.drawspecs = self.mp_draw.DrawingSpec(thickness=self.drawing_thickness, circle_radius=self.circle_radius)
        self.results = None

    def find_face(self, img, draw=True):
        '''
        :param img: Input images
        :param draw: whether to draw the tracking landmarks (Bool)
        :return: return processed img
        '''
        # convert to rgb img because the mp hands module take rgb imgs as inputs
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)

        # for face, we get the bounding box
        if self.results.multi_face_landmarks:
            for id, facelms in enumerate(self.results.multi_face_landmarks):
                if draw:
                    self.mp_draw.draw_landmarks(img, facelms, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.drawspecs, self.drawspecs)
        return img

    def get_face(self, img, lmk_no=0):
        '''
        :param img: input img
        :param lmk_no: identify a specific landmark by plotting it larger (int: 0-20 or None)
        :return:
        '''

        faces = []
        if self.results.multi_face_landmarks:
            for id, facelms in enumerate(self.results.multi_face_landmarks):
                face = []
                for lm in facelms.landmark:
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    if id == lmk_no:
                        cv2.circle(img, (x, y), 2, (255, 0, 255), cv2.FILLED)
                    face.append([id, x, y])
                faces.append(face)
        return faces


class FaceDetector:
    def __init__(self, detect_con=0.5):
        self.detect_confidence = detect_con

        self.mp_face = mp.solutions.face_detection
        self.faces = self.mp_face.FaceDetection(self.detect_confidence)
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
        self.results = self.faces.process(img_rgb)

        # for face, we get the bounding box
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if draw:
                    self.mp_draw.draw_detection(img, detection)

        return img

    def get_bounding_boxes(self, img):
        '''
        :param img: input img
        :param lmk_no: identify a specific landmark by plotting it larger (int: 0-20 or None)
        :return:
        '''

        boxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                box = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                boxes.append([id, bbox, detection.score])

                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, str(int(detection.score[0]*100))+"%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255, 2))
        return img, boxes

