from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
from OBJFileLoader import objloader
import base_modules
import math
from threading import Thread
import sys


'''
This class is adapted from some code I took from the Internet. But I closed the tab and can't remember the author. 
If this is your code, please contact me.  
I will add the source.  
'''
class Webcam:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]

    # create thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()

    def _update_frame(self):
        while(True):
            self.current_frame = self.video_capture.read()[1]

    # get the current frame
    def get_current_frame(self):
        return self.current_frame

    def write_video(self, w_cam, h_cam):
        frame_rate = 25
        resolution = (w_cam, h_cam)
        path = '../cv_videos/3dhand_control3.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(path, fourcc, frame_rate, resolution)

        return video_writer


class HandControl3D:
    def __init__(self,w_cam=1280, h_cam=720, draw_hand_tracking=True):
        self.w_cam = w_cam
        self.h_cam = h_cam
        self.draw_hand_tracking = draw_hand_tracking
        self.cap = None
        self.box = None
        self.texture_background = None
        self.angle = 0.0
        self.obj_path = None
        self.hand_tracker = base_modules.HandTracker()
        self.window_id = None
        self.obj_list = []
        self.num_objs = len(self.obj_list)

        self.display_id = -100
        self.xyz_position = [0.0, -0.5, -8]
        self.z_max = 0
        self.rotation = [0, 0, 0]
        self.obj_z_range = [5, 50]
        self.angle_add = 1.0
        self.video_writer = None

    def init_window(self, win_text="OpenGL + OpenCV"):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.w_cam, self.h_cam)
        glutInitWindowPosition(100, 100)
        self.window_id = glutCreateWindow(win_text)

    def init_gl(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)

    def init_webcam(self):
        self.cap = Webcam()
        self.cap.start()
        self.video_writer = self.cap.write_video(self.w_cam, self.h_cam)

    def init_obj(self, obj_paths=[], lowpoly_docs=[]):
        if obj_paths:
            for path in obj_paths:
                path_p = path.split("/")
                if path_p[-1] in lowpoly_docs:
                    self.box = objloader.OBJ(path, low_poly=True)
                else:
                    self.box = objloader.OBJ(path, low_poly=False)
                self.obj_list.append(self.box)
        self.num_objs = len(self.obj_list)


    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # get image from webcam
        im = self.cap.get_current_frame()
        up_img = cv2.flip(im, 1)
        image = self.hand_tracker.find_hands(up_img, draw=self.draw_hand_tracking)
        image = cv2.flip(image, 0)

        # image = self.add_buttons(image)
        tx_image = Image.fromarray(image)
        ix = tx_image.size[0]
        iy = tx_image.size[1]
        img_data = tx_image.tobytes('raw', 'BGRX', 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -10.0)
        self.draw_background()
        glPopMatrix()

        self.hand_control(up_img)
        glColor3f(1, 1, 1)

        # grab a screenshot
        buffer = glReadPixels(0, 0, self.w_cam, self.h_cam, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (self.w_cam, self.h_cam), buffer, "raw", "BGR")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = np.array(image)
        # openCV video writer expects "BGR" format
        self.video_writer.write(image)
        glutSwapBuffers()


    def draw_obj(self, obj_id=0, angle_add=1, pos=[0.5, -0.5, -0.8], z_max=140, rotation=[0, 1, 0]):
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        self.scale_and_centre(-4, 4, -2, 2, 5, z_max)
        glRotate(self.angle, rotation[0], rotation[1], rotation[2])
        self.angle += angle_add
        if self.num_objs > obj_id:
            func = self.obj_list[obj_id]
            func.render()
        else:
            self.make_cube()
        glPopMatrix()


    def hand_control(self, image):
        lm_list = self.hand_tracker.get_position(image)

        if lm_list:
            fingers = self.hand_tracker.fingers_counter(left_hand=True)
            if fingers[1] and sum(fingers) == 1:
                self.display_id = sum(fingers) - 1
                self.angle_add = 1.0
                self.xyz_position = [0.0, -1.0, -8]
                self.z_max = 1700
                self.rotation = [0, 1, 0]
                self.obj_z_range = [5, 2000]
                self.draw_obj(obj_id=self.display_id, angle_add=self.angle_add, pos=self.xyz_position,
                              z_max=self.z_max, rotation=self.rotation)

            if fingers[1] and fingers[2] and sum(fingers) == 2:
                self.display_id = sum(fingers) - 1
                self.angle_add = 1.0
                self.xyz_position = [0.0, -0.5, -8]
                self.z_max = 150
                self.rotation = [0, 1, 1]
                self.obj_z_range = [5, 400]
                self.draw_obj(obj_id=self.display_id, angle_add=self.angle_add, pos=self.xyz_position,
                              z_max=self.z_max, rotation=self.rotation)

            if fingers[1] and fingers[2] and fingers[3] and fingers[4] and sum(fingers) == 5:
                self.display_id= 100000
                self.angle_add = 1.0
                self.xyz_position = [0.0, 1.0, -5.5]
                self.z_max = 30
                self.obj_z_range = [10, 100]
                self.rotation = [0, 1, 1]
                self.draw_obj(obj_id=self.display_id, angle_add=self.angle_add, pos=self.xyz_position,
                              z_max=self.z_max, rotation=self.rotation)

            if not self.display_id == -100:
                if fingers[0] and fingers[1] and fingers[2] and sum(fingers) == 3:
                    thumb1, thumb2 = lm_list[4][1:]
                    index1, index2 = lm_list[8][1:]

                    length = int(math.hypot(int(index1) - int(thumb1), int(index2) - int(thumb2)))
                    # length range between index finger and thumb: 15 - 335
                    hand_range = [30, 315]
                    size = int(np.interp(length, hand_range, [self.obj_z_range[1], self.obj_z_range[0]]))
                    self.z_max = size

                self.draw_obj(obj_id=self.display_id, angle_add=self.angle_add, pos=self.xyz_position,
                              z_max=self.z_max, rotation=self.rotation)

    def draw_background(self):
        # draw background
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(4.0, 3.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-4.0, 3.0, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def scale_and_centre(self, xmin, xmax, ymin, ymax, zmin, zmax):
        xs = (xmax - xmin) / 2.0
        ys = (ymax - ymin) / 2.0
        zs = (zmax - zmin) / 2.0

        array1 = np.array([xs, ys, zs])
        qs = array1.max()
        sk = 1.0 / qs
        glScalef(sk, sk, sk)

    def make_cube(self):
        # RENDER OBJECT
        glTranslate(0, 0, -10)
        glRotate(self.angle, 1, 1, 1)
        self.angle += self.angle_add

        # 8 vertices for a cube
        vertices = ((1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1))
        # we have 12 edges
        edges = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7))
        # 6 surfaces
        surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))

        # make lines
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        # make surfaces
        glBegin(GL_QUADS)
        for surface in surfaces:
            for vertex in surface:
                glColor3fv((0, 1, 0))
                glVertex3fv(vertices[vertex])
        glEnd()


def main():
    # ----------------------
    w_cam, h_cam = 1280, 720
    # ----------------------

    control = HandControl3D(w_cam, h_cam, True)

    control.init_window(win_text="OpenGL + OpenCV")

    path = [ '3d_objs/bank.obj',
             '3d_objs/10870_turkey_leg_v1_L3.obj']
    control.init_obj(obj_paths=path, lowpoly_docs=["bank.obj"])

    control.init_webcam()

    def keyboard(key, x, y):
        # Allow to quit by pressing 'Esc' or 'q'
        # I'm missing some functions in the library so I can't really exit the loop
        if key == b'a':
            sys.exit()
        if key == b'q':
            control.video_writer.release()

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(control.render_scene)
    glutIdleFunc(control.render_scene)
    # the following function doesn't exist in my library
    # glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
    glutKeyboardFunc(keyboard)

    control.init_gl()
    glutMainLoop()


if __name__ == "__main__":
    main()