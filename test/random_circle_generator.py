import random
from pathlib import Path
import numpy as np
import cv2
import os


def create_random_dataset(image_width, image_height, N_IMAGES, output_path):
    output_path = Path(output_path, "output")
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, N_IMAGES):
        img = np.zeros(shape=(image_height, image_width, 3), dtype=np.int16)
        N_CIRCLES = random.randint(15, 35)

        for j in range(0, N_CIRCLES):
            circle_center = (random.randint(0, image_width), random.randint(0, image_height))
            radius = random.randint(10, 15)
            img = cv2.circle(img, circle_center, radius, (255, 255, 255), -1)

        cv2.imwrite(
            str(Path(output_path, f"img_{i:03d}.png")), img
        )

image_width = 252
image_height = 1024
path = Path(os.getcwd())
create_random_dataset(image_width, image_height, 4, path)
