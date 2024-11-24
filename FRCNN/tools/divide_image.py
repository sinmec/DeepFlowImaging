import numpy as np
import math

def divide_image(img, resized_img_size):
    h, w = img.shape[:2]

    columns = math.ceil(w / resized_img_size)
    lines = math.ceil(h / resized_img_size)

    x = np.linspace(resized_img_size / 2, w - resized_img_size / 2, num=columns)
    y = np.linspace(resized_img_size / 2, h - resized_img_size / 2, num=lines)

    subdivided_images = []
    pos = []

    for i in x:
        for j in y:
            x1, x2 = int(i - resized_img_size / 2), int(i + resized_img_size / 2)
            y1, y2 = int(j - resized_img_size / 2), int(j + resized_img_size / 2)

            crop = img[y1:y2, x1:x2] / 255.0
            pos.append([y1, y2, x1, x2])
            subdivided_images.append(crop)

    return subdivided_images, pos
