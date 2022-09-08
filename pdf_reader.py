import cv2 as cv
import numpy as np
from scipy.interpolate import UnivariateSpline
import fitz
import os
from matplotlib import pyplot as plt
from collections import Counter
from fpdf import FPDF


DATA_PATH = "pdf_data"


class PDFReader:

    def __init__(self):
        self.increase_lut = self.create_lut([0, 64, 128, 192, 255], [0, 70, 140, 210, 255])
        self.decrease_lut = self.create_lut([0, 64, 128, 192, 255], [0, 56, 118, 175, 244])
        self.images = []
        self.dis_images = []
        self.ec_images = None
        self.ec_on = False

        self.bg_mask = []
        self.bg_smooth_mask = []
        self.fg_mask = []
        self.fg_smooth_mask = []
        self.bg_color = None
        self.fg_color = None

    def create_lut(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    def open_pdf(self, file_name):
        self.cur_path = os.path.join(DATA_PATH, file_name[:-4])
        self.file_name = file_name

        zoom_x = 4.0
        zoom_y = 4.0
        mat = fitz.Matrix(zoom_x, zoom_y)
        doc = fitz.open('pdfs/' + file_name)

        if not os.path.exists(self.cur_path):
            os.mkdir(self.cur_path)
        img0 = self.cur_path + "/%s-page-0.jpg" % (file_name[:-4])
        if not os.path.exists(img0):
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                pix.save(self.cur_path + "/%s-page-%i.jpg" % (file_name[:-4], page.number))

        # m = [page.get_pixmap(matrix=mat) for page in doc]
        self.images = [cv.imread(self.cur_path + "/%s-page-%i.jpg" % (file_name[:-4], page.number)) for page in doc]
        self.dis_images = self.images

        # Preprocess images
        self.bg_mask = [0] * len(self.images)
        self.fg_mask = [0] * len(self.images)
        self.bg_smooth_mask = [0] * len(self.images)
        self.fg_smooth_mask = [0] * len(self.images)

        for i in range(len(self.images)):
            self.preprocess(i)

    def save_pdf(self):
        pdf = FPDF()

        for i, image in enumerate(self.dis_images):
            file = 'temp-save-img' + str(i) + '.png'
            cv.imwrite(file, image)
            pdf.add_page()
            pdf.image(file, 0, 0, 210, 297)
            os.remove(file)

        pdf.output('pdfs/' + self.file_name[:-4] + '-edited3.pdf', 'F')
        pdf.close()


    def preprocess(self, idx):

        image = self.images[idx]
        print(idx)
        # background masks
        rng = 30
        target = 255 - rng

        b, g, r = cv.split(image)
        hash = b + g * 256 + r * 65536
        vals = np.squeeze(hash.reshape(1, -1)).tolist()
        pix = Counter(vals).most_common(1)[0][0]

        B = pix % 256
        pix = pix // 256
        G = pix % 256
        pix = pix // 256
        R = pix % 256

        print("background:")
        print(R, G, B)
        self.ini_bg_color = [B, G, R]
        _b = (b + max(0, (target - B))) % 256
        _g = (g + max(0, (target - G))) % 256
        _r = (r + max(0, (target - R))) % 256

        conv_image = cv.merge((_b, _g, _r)).astype(np.uint8)
        gray = cv.cvtColor(conv_image, cv.COLOR_RGB2GRAY)

        thresh = max([min([B, G, R]), target]) - min([B, G, R, rng]) - 1
        print(thresh)
        ret, self.bg_mask[idx] = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
        self.bg_smooth_mask[idx] = cv.dilate(self.bg_mask[idx], np.ones((3, 3), np.uint8)) - self.bg_mask[idx]

        # foreground masks
        rng = 60
        target = 255 - rng

        self.fg_mask[idx] = (255*np.ones_like(self.bg_mask[idx]) - self.bg_mask[idx]).astype(np.uint8)
        res = (255*np.ones_like(self.bg_mask[idx]) - self.bg_mask[idx]).astype(np.uint8)

        blur = cv.GaussianBlur(res, (5, 5), 0)
        # self.sshow("blur", blur)
        ret, th = cv.threshold(blur, 25, 255, cv.THRESH_BINARY)
        # self.sshow('th', th)
        dilated = cv.dilate(th, np.ones((4, 4), np.uint8), iterations=2)
        # self.sshow('dil', dilated)
        _, contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        ca_thresh = 10**5
        ca_thresh2 = 10**6

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        for i, c in enumerate(contours):
            if cv.contourArea(c) < ca_thresh or cv.contourArea(c) > ca_thresh2:
                continue
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
            self.fg_mask[idx][int(boundRect[i][1]): int(boundRect[i][1] + boundRect[i][3]), int(boundRect[i][0]): int(boundRect[i][0] + boundRect[i][2])] = 0
            self.bg_mask[idx][int(boundRect[i][1]): int(boundRect[i][1] + boundRect[i][3]), int(boundRect[i][0]): int(boundRect[i][0] + boundRect[i][2])] = 0

        # drawing = np.zeros((gg.shape[0], gg.shape[1], 3), dtype=np.uint8)
        #
        # for i in range(len(contours)):
        #     if cv.contourArea(contours[i]) < ca_thresh or cv.contourArea(contours[i]) > ca_thresh2:
        #         continue
        #     color = (255, 0, 0)
        #     cv.drawContours(drawing, contours_poly, i, color)
        #     cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #                  (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        #

        # self.sshow('Contours', drawing)
        # for cnt in c:
        #     if cv.contourArea(cnt) >= ca_thresh:
        #         epsilon = 0.1 * cv.arcLength(cnt, True)
        #         approx = cv.approxPolyDP(cnt, epsilon, True)
        #         print(type(approx))
        #         print(approx)
        # print(res)
        # print(type(res))
        # cv.drawContours(gg, c, -1, (0, 255, 0), 2)
        # # print(res.dtype)
        # self.sshow("inter", gg)
        # cv.waitKey(0)
        
        # self.sshow("bg", self.bg_mask[idx])
        # self.sshow("fg", self.fg_mask[idx])

        res = image[self.fg_mask[idx] == 255, :]
        b = res[:, 0]
        g = res[:, 1]
        r = res[:, 2]
        hash = b + g * 256 + r * 65536
        vals = np.squeeze(hash.reshape(1, -1)).tolist()
        pix = Counter(vals).most_common(1)[0][0]

        B = pix % 256
        pix = pix // 256
        G = pix % 256
        pix = pix // 256
        R = pix % 256

        print("foreground:")
        print(R, G, B)
        self.fg_color = None
        b, g, r = cv.split(image)
        _b = (b + max(0, (target - B))) % 256
        _g = (g + max(0, (target - G))) % 256
        _r = (r + max(0, (target - R))) % 256

        conv_image = cv.merge((_b, _g, _r)).astype(np.uint8)
        # self.sshow("changed images", conv_image)
        gray = cv.cvtColor(conv_image, cv.COLOR_RGB2GRAY)
        gray[self.fg_mask[idx] == 0] = 0
        # self.sshow("gray", gray)

        thresh = max([min([B, G, R]), target]) - min([B, G, R, rng]) - 1
        print(thresh)
        ret, self.fg_mask[idx] = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
        # self.fg_mask[idx] = cv.dilate(self.fg_mask[idx], np.ones((3, 3), np.uint8))
        self.fg_smooth_mask[idx] = cv.dilate(self.fg_mask[idx], np.ones((3, 3), np.uint8)) - self.fg_mask[idx]
        # self.sshow("fg after", self.fg_mask[idx])
        # cv.waitKey(0)

    def sshow(self, txt, img):
        img = cv.resize(img, None, fx=0.3, fy=0.3)
        cv.imshow(txt, img)

    def get_images(self):
        if self.ec_on:
            if self.ec_images is None:
                self.ec_images = [img.copy() for img in self.dis_images]
                for i in range(len(self.ec_images)):
                    self.eye_comfort(i)
                    print(np.all(self.ec_images[i] == self.dis_images[i]))
            print("eye comfort")
            # print(len(self.ec_images))
            # print(len(self.dis_images))
            return [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in self.ec_images]
        print("normal")
        return [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in self.dis_images]

    def save_mark(self, x):
        path = DATA_PATH + "/" + self.file_name[:-4]
        print(path)
        f = open(os.path.join(path, "marks.txt"), 'w')
        f.write(str(x))
        f.close()

    def get_mark(self):
        path = DATA_PATH + "/" + self.file_name[:-4]
        ff = os.path.join(path, "marks.txt")
        if not os.path.exists(ff): return -1
        print(path)
        f = open(ff, 'r')
        x = f.read()
        f.close()
        return int(x)

    def eye_comfort(self, idx):
        image = self.ec_images[idx]

        b, g, r = cv.split(image)

        b = cv.LUT(b, self.decrease_lut).astype(np.uint8)
        b = cv.LUT(b, self.decrease_lut).astype(np.uint8)

        g = cv.LUT(g, self.decrease_lut).astype(np.uint8)
        r = cv.LUT(r, self.increase_lut).astype(np.uint8)
        image = cv.merge((b, g, r))

        h, s, v = cv.split(cv.cvtColor(image, cv.COLOR_RGB2HSV))
        s = cv.LUT(s, self.increase_lut).astype(np.uint8)
        self.ec_images[idx] = cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2RGB)

    def set_ec(self, value):
        self.ec_on = value

    def change_color(self, bg_color, text_color):
        self.ec_images = None
        if bg_color is not None:
            self.bg_color = bg_color
        if text_color is not None:
            self.fg_color = text_color

        self.dis_images = [img.copy() for img in self.images]
        for i in range(len(self.dis_images)):
            if self.bg_color is not None:
                self.change_background_color(i)
            if self.fg_color is not None:
                self.change_text_color(i)


    def change_background_color(self, idx):
        image = self.dis_images[idx]
        new_color = self.bg_color

        image[self.bg_mask[idx] == 255, :] = new_color

        image = image.astype(np.float)
        image[self.bg_smooth_mask[idx] == 255, :] *= [0.7] * 3
        image[self.bg_smooth_mask[idx] == 255, :] += (0.3 * np.array(new_color)).tolist()
        self.dis_images[idx] = image.astype(np.uint8)

    def change_text_color(self, idx):
        image = self.dis_images[idx]
        new_color = self.fg_color

        image[self.fg_mask[idx] == 255, :] = new_color

        image = image.astype(np.float)
        image[self.fg_smooth_mask[idx] == 255, :] *= [0.4] * 3
        image[self.fg_smooth_mask[idx] == 255, :] += \
            (0.4 * np.array(self.bg_color if self.bg_color is not None else self.ini_bg_color)).tolist()
        image[self.fg_smooth_mask[idx] == 255, :] += (0.2 * np.array(new_color)).tolist()
        self.dis_images[idx] = image.astype(np.uint8)

    def run(self):

        # bg_c = [int(w) for w in input("Back Ground Color: ").split(' ')]
        # text_c = [int(w) for w in input("Text Color: ").split(' ')]
        # bg_c = []
        # text_c = []
        # if len(bg_c) != 3: bg_c = [255, 255, 0]
        # if len(text_c) != 3: text_c = [0, 0, 255]

        self.open_pdf('4.pdf')
        images = self.get_images()
        normal_image = images[-1]
        self.change_color([0, 255, 255], None)
        images = self.get_images()
        bg_image = images[-1]
        self.change_color(None, [0, 0, 255])
        images = self.get_images()
        fg_image = images[-1]

        # self.save_pdf()
        plt.subplot(231)
        plt.imshow(normal_image)
        plt.xticks([]), plt.yticks([])
        plt.title("original")
        plt.subplot(234)
        plt.imshow(fg_image)
        plt.xticks([]), plt.yticks([])
        plt.title('colored')

        plt.subplot(232)
        plt.imshow(self.fg_mask[-1], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title('fg_mask')
        plt.subplot(235)
        plt.imshow(self.fg_smooth_mask[-1], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title("fg_smooth_mask")

        plt.subplot(233)
        plt.imshow(self.bg_mask[-1], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title("bg_mask")
        plt.subplot(236)
        plt.imshow(self.bg_smooth_mask[-1], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title("bg_smooth_mask")


        plt.show()


if __name__ == '__main__':

    pdfreader = PDFReader()
    pdfreader.run()
