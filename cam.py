import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import threading

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_righteye_2splits.xml')

#
# def get_eyes(eyes_rec, image):
#     # return eye_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)
#     faces = face_cascade.detectMultiScale(image, 1.3, 5)
#
#     mx_area = 0
#     face = ()
#     for (x, y, w, h) in faces:
#         if mx_area < w*h:
#             mx_area = w*h
#             face = (x, y, w, h)
#
#     if mx_area == 0:
#         return []
#     # print("face")
#
#     (x, y, w, h) = face
#     roi = gray[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi)
#
#     print(len(eyes))
#     if len(eyes) > 0:
#         eyes.sort()
#         eyes[:, 0] += x
#         eyes[:, 1] += y
#         eyes_rec[0] = eyes[0]
#     if len(eyes) > 1:
#         eyes_rec[1] = eyes[1]
#
#     return eyes
#
#
# def print_eyes(image, eyes):
#     for (ex, ey, ew, eh) in eyes:
#         cv.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


def f(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv.merge((cl, a, b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final

#
# class mythread(threading.Thread):
#
#
#     def __init__(self, cap):
#         threading.Thread.__init__(self)
#         self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
#         self.cp = CamProc()
#         self.cap = cap
#         ret, frame = self.cap.read()
#
#     def run(self):
#
#         # cap = cv.VideoCapture(0)
#         cap = self.cap
#
#         if not cap.isOpened():
#             print("Camera didn't open")
#             exit(0)
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("video end")
#                 break
#
#             gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
#             faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]
#                 eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.2, 7)
#                 dis = []
#                 for eye in eyes:
#                     (ex, ey, ew, eh) = eye
#                     roi = roi_color[ey:ey + eh, ex:ex + ew]
#                     roig = roi_gray[ey:ey + eh, ex:ex + ew]
#                     roig = cv.GaussianBlur(roig, (5, 5), 0)
#                     roig = 255 - roig
#                     center = (roi.shape[1] / 2, roi.shape[0] / 2)
#
#                     circles = cv.HoughCircles(roig, cv.HOUGH_GRADIENT, 1, 100,
#                                               param1=130, param2=12, minRadius=5, maxRadius=12)
#
#                     if circles is not None:
#                         rc = circles[0, 0]
#                         circles = np.uint16(np.around(circles))
#                         for c in circles[0, :]:
#                             cv.circle(roi, (c[0], c[1]), c[2], (255, 0, 0))
#                             cv.circle(roi, (c[0], c[1]), 2, (0, 255, 0))
#                             dis.append(rc[0] - center[0])
#                             break
#
#                 if type(eyes) == np.ndarray:
#                     self.cp.feed(True)
#                 else:
#                     self.cp.feed(False)
#
#                 for (ex, ey, ew, eh) in eyes:
#                     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#         cap.release()

def test():

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Camera didn't open")
        exit(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("video end")
            break

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 18)
            dis = []
            for eye in eyes:
                (ex, ey, ew, eh) = eye
                roi = roi_color[ey:ey + eh, ex:ex + ew]
                roig = roi_gray[ey:ey + eh, ex:ex + ew]
                roig = cv.GaussianBlur(roig, (5, 5), 0)
                roig = 255 - roig
                center = (roi.shape[1]/2, roi.shape[0]/2)

                # _, roig = cv.threshold(roig, 45, 255, cv.THRESH_BINARY_INV)
                circles = cv.HoughCircles(roig, cv.HOUGH_GRADIENT, 1, 100,
                                          param1=125, param2=13, minRadius=5, maxRadius=12)

                if circles is not None:
                    rc = circles[0,0]
                    circles = np.uint16(np.around(circles))
                    for c in circles[0,:]:
                        cv.circle(roi, (c[0],c[1]), c[2], (255, 0, 0))
                        cv.circle(roi, (c[0],c[1]), 2, (0, 255, 0))
                        dis.append(rc[0] - center[0])
                        break

                cv.circle(roi, (roi.shape[1]//2, roi.shape[0]//2), 7, (100, 0, 0))



                # _, contours, _ = cv.findContours(roig, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
                # for cnt in contours:
                #     (x, y, w, h) = cv.boundingRect(cnt)
                #     # cv.drawContours(roi, cnt, -1, (0, 0, 255), 1)
                #     cv.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
                #     break

                # roi = cv.resize(roi, None, fx=2.5, fy=2.5)
                # roig = cv.resize(roig, None, fx=2.5, fy=2.5)
                # cv.imshow('gray', roig)
                # cv.imshow('eyes', roi)
            #
            if len(dis) > 0:
                mx = max(dis)
                mn = min(dis)
                # print(dis)
                th = 7
                th2 = 10
                if mx > th and mx < th2: print("left")
                elif mn < -th and mn > -th2: print('right')

            if type(eyes) == np.ndarray:
                cp.feed(True)
            else:
                cp.feed(False)

            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv.imshow('gray', gray)
        cv.imshow('eyes', frame)

        if cv.waitKey(30) == 27:
            break

    cv.destroyAllWindows()
    cap.release()
    exit(0)

class CamProc:
    sz = 100
    restT = 6
    openT = 4
    closeT = 4


    def __init__(self):
        self.h = []

    def ridx(self, skip, x):
        if skip >= len(self.h): return -1
        for i in reversed(range(skip+1)):
            if self.h[i] == x:
                return i
        return -1

    def feed(self, v):
        # print(v)
        self.h.append(v)
        if len(self.h) > self.sz:
            del self.h[0]

        if v:
            z1 = self.ridx(len(self.h)-1, 0)
            if z1 >= 0 and len(self.h) - z1 - 1 == self.restT:
                o1 = self.ridx(z1, 1)
                if o1 >= 0 and z1 - o1 <= self.closeT:
                    z2 = self.ridx(o1, 0)
                    if z2 >= 0 and o1 - z2 <= self.openT:
                        o2 = self.ridx(z2, 1)
                        if o2 >= 0 and z2 - o2 <= self.closeT:
                            z3 = self.ridx(o2, 0)
                            if o2 - z3 >= self.restT:
                                print("2 wings")
                            elif o2 - z3 <= self.openT:
                                o3 = self.ridx(z3, 1)
                                if z3 - o3 <= self.closeT:
                                    z4 = self.ridx(o3, 0)
                                    if o3 - z4 >= self.restT:
                                        print('3 wings')



if __name__ == "__main__":
    # thread = mythread()
    # thread.start()
    # while True:
    #     pass
    cp = CamProc()
    test()
    exit(0)
    video = "http://192.168.43.1:4747/mjpegfeed"
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Camera didn't open")
        exit(0)

    eyes_rec = [(), ()]
    i = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("video end")
            break

        if cv.waitKey(1) == 27:
            break

        eyes = []
        # frame = cv.rotate(frame, cv.ROTATE_180)
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # if i%10 == 0:
        # lb = f(frame)

        # gray = cv.medianBlur(gray, 3)
        # gray = cv.equalizeHist(gray)
        # gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
        eyes = get_eyes(eyes_rec, frame)
        cv.imshow('gray', gray)

        print_eyes(frame, eyes)
        if len(eyes_rec[0]) > 0:
            (x, y, w, h) = eyes_rec[0]
            cv.imshow('right eye', cv.resize(gray[y:y + h, x:x + w], (200, 150)))
        if len(eyes_rec[1]) > 1:
            (x, y, w, h) = eyes_rec[1]
            cv.imshow('left eye', cv.resize(gray[y:y + h, x:x + w], (200, 150)))
        cv.imshow('eyes', frame)


        # cv.imshow("face", frame)

        # i = i+1
    cv.destroyAllWindows()
    cap.release()


    # img = cv.imread('my_face.jpg')
    # img = cv.resize(img, None, fx=0.6, fy=0.6)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #
    # eyes = get_eyes(gray)
    # print_eyes(img, eyes)
    #
    # cv.imshow('img', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()