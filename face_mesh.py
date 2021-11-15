import cv2
import mediapipe as mp


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


def main():
    cap = cv2.VideoCapture(0)

    face_det = FaceMesh(drawing_thickness=1, cir_radius=1)

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)

        img = face_det.find_face(img)
        lm_list = face_det.get_face(img)
        # print(lm_list)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()