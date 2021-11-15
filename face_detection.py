import cv2
import mediapipe as mp


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


def main():
    cap = cv2.VideoCapture(0)

    face_det = FaceDetector()

    while True:
        success, img = cap.read()
        # take the mirrored img if needed
        img = cv2.flip(img, 1)

        img = face_det.find_pose(img)
        img, boxes = face_det.get_bounding_boxes(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()