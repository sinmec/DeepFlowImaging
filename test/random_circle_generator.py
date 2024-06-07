import random
from pathlib import Path
import numpy as np
import cv2
import os
import h5py


# ellipse_bounds = [ellipse_width_bounds, ellipse_height_bounds, ellipse_ahgle_bounds]
# ellipse_bounds = [(x, y), (w, z), (a, b)]
def create_random_dataset(image_width, image_height, N_IMAGES, output_path, ellipse_bounds):
    output_path = Path(output_path, "output")
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, N_IMAGES):
        contours = []

        img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        N_ELLIPSES = random.randint(15, 35)

        for j in range(0, N_ELLIPSES):
            ellipse_center = (random.randint(0, image_width), random.randint(0, image_height))
            ellipse_width = random.randint(ellipse_bounds[0][0], ellipse_bounds[0][1])
            ellipse_height = random.randint(ellipse_bounds[1][0], ellipse_bounds[1][1])
            ellipse_angle = random.randint(ellipse_bounds[2][0], ellipse_bounds[2][1])

            img = cv2.ellipse(img, ellipse_center,
                              (ellipse_width, ellipse_height),
                              ellipse_angle, 0, 360, (255, 255, 255), -1)

            ellipse_contour = ellipse_to_contour(ellipse_center, ellipse_width, ellipse_height, ellipse_angle)
            contours.append(ellipse_contour)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        cv2.imwrite(
            str(Path(output_path, f"{i:03d}.png")), img
        )

        print(N_ELLIPSES, len(contours))

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


def ellipse_to_contour(center, width, height, angle):
    ellipse_poly = cv2.ellipse2Poly(center, (width, height), int(angle), 360, 1, 1)

    N_points = len(ellipse_poly)
    cv2_contour = np.zeros((N_points, 1, 2), dtype=int)

    for i, (x, y) in enumerate(ellipse_poly):
        cv2_contour[i, 0, 0] = ellipse_poly[i][0]
        cv2_contour[i, 0, 1] = ellipse_poly[i][1]

    return cv2_contour


image_width = 252
image_height = 1024
path = Path(os.getcwd())
ellipse_bounds = [[15, 20], [15, 20], [0, 30]]
create_random_dataset(image_width, image_height, 10, path, ellipse_bounds)
