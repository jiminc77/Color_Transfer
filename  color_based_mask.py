import cv2
import numpy as np
import os

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 정의한 색상 범위에 따라 마스크 생성
    lower_hue = np.array([0, 0, 0]) 
    upper_hue = np.array([100, 100, 100])
    mask = cv2.inRange(image_hsv, lower_hue, upper_hue)
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


input_folder = "/home/jovyan/Color_Transfer/examples/content"
output_folder = "/home/jovyan/Color_Transfer/examples/content_segment"

image_files = os.listdir(input_folder)

for image_file in image_files:

    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    segmented = segment_plant(image)

    sharpened = sharpen_image(segmented)
    output_path = os.path.join(output_folder, 'black_.png')

    cv2.imwrite(output_path, sharpened)
