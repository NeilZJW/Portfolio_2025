# -*- coding: utf-8 -*-
# author: Neil

import cv2
import numpy as np
import os


class CheckImage:
    def __init__(self, image_path, target_path="images/0.png"):
        self.image_path = image_path
        self.target_path = target_path
        self.scale_img, self.scale_tar, self.color_img, self.color_tar = self._get_feature()

    def _get_feature(self):
        image = cv2.imread(self.image_path)
        tar_image = cv2.imread(self.target_path)
        scale_img = image.shape[:2]
        scale_tar = tar_image.shape[:2]
        color_img = image.shape[-1]
        color_tar = tar_image.shape[-1]
        return scale_img, scale_tar, color_img, color_tar

    def check_size(self):
        h, w = self.scale_img
        if (h, w) != self.scale_tar:
            if h > self.scale_tar[1] or w > self.scale_tar[0]:
                # zoom out
                interpolation = cv2.INTER_AREA
                return interpolation
        else:
            # zoom in
            # qulity: cv2.INTER_CUBIC
            # speed: cv2.INTER_LINEAR
            interpolation = cv2.INTER_CUBIC
            return interpolation

    def check_color(self):
        if self.color_img != self.color_tar:
            if self.color_img == 1:
                # convert to color
                g_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                c_image = cv2.cvtColor(g_image, cv2.COLOR_GRAY2BGR)
                return c_image
        elif self.color_img == 4:
            rgba_image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2BGR)
            return rgb_image
        else:
            print("Unknown color of image!")


def adjust_face(image_path, target_height, target_width):
    global flag
    flag = False
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No face detected")
        return flag
    elif len(faces) > 1:
        print("ERROR! More than one face detected")
        return flag
    else:
        flag = True
        x, y, w, h = faces[0]

        ori_height, ori_width = image.shape[:2]
        ratio = ori_width / ori_height
        target_ratio = target_width / target_height

        if ratio > target_ratio:
            new_width = target_width
            new_height = int(new_width / ratio)
            print("1", (new_height, new_width))
        else:
            new_height = target_height
            new_width = int(new_height * ratio)
            print("2", (new_height, new_width))

        # resize
        interpolation = CheckImage(image_path).check_size()
        re_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        # print("resize: {}".format(re_image.shape))

        # fill
        if new_width != target_width or new_height != target_height:
            # create new background
            final_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            final_image.fill(255)

            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            final_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = re_image
        else:
            final_image = re_image
    return final_image


input_dir = "images/input/"
output_dir = "images/output/"
for img in os.listdir(input_dir):
    image_path = input_dir + img
    print("original image: {}".format(cv2.imread(image_path).shape))
    target_height, target_width = CheckImage(image_path).scale_tar
    flag = False
    # print((target_height, target_width))
    final_image = adjust_face(image_path, target_height, target_width)
    if flag:
        print("shape: {}".format(final_image.shape))
        cv2.imwrite(output_dir + img.split(".")[0] + ".png", final_image)
        # cv2.imshow("final image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
