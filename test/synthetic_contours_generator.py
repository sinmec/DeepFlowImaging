import random
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def create_random_dataset(
        h5_file_name, image_width, image_height, N_IMAGES, output_path, ellipse_options
):
    output_path = Path(output_path, "output")
    output_path.mkdir(parents=True, exist_ok=True)
    img_contours = []
    img_collection = []
    MAX_ITERATIONS = 500

    failed_images = 0
    for i in tqdm(range(0, N_IMAGES), desc="Creating synthetic ellipse images"):
        contours = []

        img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        img_check = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        N_ELLIPSES = random.randint(
            ellipse_options["N_ellipses"]["min"], ellipse_options["N_ellipses"]["max"]
        )
        valid_ellipses = 0
        it = 0

        failed_ellipses = 0
        while valid_ellipses < N_ELLIPSES:
            if it > MAX_ITERATIONS:
                failed_ellipses += 1
                break

            ellipse_axis_1 = random.randint(
                ellipse_options["width"]["min"], ellipse_options["width"]["max"]
            )
            ellipse_axis_2 = random.randint(
                ellipse_options["width"]["min"], ellipse_options["width"]["max"]
            )
            ellipse_angle = random.randint(
                ellipse_options["angle"]["min"], ellipse_options["angle"]["max"]
            )

            major_axis = max(ellipse_axis_1, ellipse_axis_2)
            minor_axis = min(ellipse_axis_1, ellipse_axis_2)

            BOUNDARY_OFFSET = 2
            max_dimension = max(major_axis, minor_axis) + BOUNDARY_OFFSET
            ellipse_center = (
                random.randint(max_dimension, image_width - max_dimension),
                random.randint(max_dimension, image_height - max_dimension),
            )

            check = np.zeros_like(img)
            empty_img = np.zeros_like(img)

            new_ellipse = cv2.ellipse(
                empty_img,
                ellipse_center,
                (major_axis, minor_axis),
                ellipse_angle,
                0,
                360,
                (255, 255, 255),
                -1,
            )

            check[new_ellipse == 255] += 1
            check[img_check == 255] += 1

            if np.max(check) < 2:
                valid_ellipses += 1

                img = cv2.ellipse(
                    img,
                    ellipse_center,
                    (major_axis, minor_axis),
                    ellipse_angle,
                    0,
                    360,
                    (255, 255, 255),
                    ellipse_options["line_width"],
                )

                img_check = cv2.ellipse(
                    img_check,
                    ellipse_center,
                    (major_axis, minor_axis),
                    ellipse_angle,
                    0,
                    360,
                    (255, 255, 255),
                    -1,
                )

                ellipse_contour = ellipse_to_contour(
                    ellipse_center, major_axis, minor_axis, ellipse_angle
                )
                contours.append(ellipse_contour)

            else:
                it += 1
                pass
        if failed_ellipses:
            print(
                f"\tWarning: At image {i} a total of {failed_ellipses} ellipses failed."
                f"\n\tPlease check the input parameters"
            )

            failed_images += 1
        cv2.imwrite(str(Path(output_path, f"{i:03d}.png")), img)

        img_contours.append(contours)
        img_collection.append(img)

    if failed_images:
        print(
            f"Warning: {failed_images}/{N_IMAGES} images resulted in ellipse fitting problems.\n"
            f"Please check the input parameters."
        )

    export_h5(output_path, h5_file_name, img_collection, img_contours)


def export_h5(output_path, h5_file_name, img_collection, img_contours):
    h5_file_path = Path(output_path, h5_file_name)
    with h5py.File(h5_file_path, "w") as h5_file:
        for img_index, (img, contours) in enumerate(zip(img_collection, img_contours)):
            img_group_name = f"img_{img_index:03d}"
            img_group = h5_file.create_group(img_group_name)

            img_group.create_dataset("img", data=img, compression="gzip")

            contours_group = img_group.create_group("contours")
            for idx, contour in enumerate(contours):
                contours_group.create_dataset(f"cnt_{idx:06d}", data=contour)
    h5_file.close()


def ellipse_to_contour(center, width, height, angle):
    ellipse_poly = cv2.ellipse2Poly(center, (width, height), int(angle), 360, 1, 1)

    N_points = len(ellipse_poly)
    cv2_contour = np.zeros((N_points, 1, 2), dtype=int)

    for i, (x, y) in enumerate(ellipse_poly):
        cv2_contour[i, 0, 0] = ellipse_poly[i][0]
        cv2_contour[i, 0, 1] = ellipse_poly[i][1]

    return cv2_contour
