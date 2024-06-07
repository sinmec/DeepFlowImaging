import random
from pathlib import Path
import numpy as np
import cv2
import os
import h5py


def create_random_dataset(image_width, image_height, N_IMAGES, output_path):
    output_path = Path(output_path, "output")
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, N_IMAGES):
        img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        N_ELLIPSES = random.randint(15, 35)

        for j in range(0, N_ELLIPSES):
            ellipse_center = (random.randint(0, image_width - 1), random.randint(0, image_height - 1))
            ellipse_width = random.randint(15, 20)
            ellipse_height = random.randint(15, 20)
            ellipse_angle = random.randint(0, 30)

            img = cv2.ellipse(img, ellipse_center,
                              (ellipse_width, ellipse_height),
                              ellipse_angle, 0, 360, (255, 255, 255), -1)

        # Converter a imagem para escala de cinza
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imwrite(
            str(Path(output_path, f"img_{i:03d}.png")), img
        )

        export_h5(output_path, img, contours, i)


def export_h5(output_path, img, contours, img_index):
    h5_file_name = "test.h5"
    h5_file_path = Path(output_path, h5_file_name)

    with h5py.File(h5_file_path, "a") as h5_file:
        img_group_name = f"img_{img_index:03d}"
        img_group = h5_file.create_group(img_group_name)

        img_group.create_dataset("img", data=img)

        contours_group = img_group.create_group("contours")
        for idx, contour in enumerate(contours):
            contours_group.create_dataset(f"cnt_{idx:06d}", data=contour)


image_width = 252
image_height = 1024
path = Path(os.getcwd())
create_random_dataset(image_width, image_height, 4, path)
