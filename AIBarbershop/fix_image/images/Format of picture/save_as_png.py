import cv2
import os


input_dir = "./img/"
output_dir = "../input/"
for img in os.listdir(input_dir):
    image = cv2.imread(input_dir + img)
    cv2.imwrite(output_dir + img.split(".")[0] + ".png", image)
