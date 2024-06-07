import random
from pathlib import Path
import numpy as np
import cv2
import os


def create_random_dataset(image_width, image_height, N_IMAGES, output_path):
    output_path = Path(output_path, "output")
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, N_IMAGES):
        img = np.zeros((image_height, image_width), dtype=np.uint8)
        N_ELLIPSES = random.randint(15, 35)

        for j in range(0, N_ELLIPSES):
            ellipse_center = (random.randint(0, image_width), random.randint(0, image_height))
            ellipse_width = random.randint(15, 20)
            ellipse_height = random.randint(15, 20)
            ellipse_angle = random.randint(0, 30)

            img = cv2.ellipse(img, ellipse_center,
                        (int(ellipse_width), int(ellipse_height)),
                        ellipse_angle, 0, 360, (255, 255, 255), -1)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imwrite(
            str(Path(output_path, f"img_{i:03d}.png")), img
        )


def export_h5():
    h5_file_name = f"{image_folder_name}_{self.number_of_exports:06d}.h5"
    h5_file_path = Path(self.output_path, h5_file_name)

    h5_file = h5py.File(h5_file_path, "w")
    h5_file.attrs['date'] = datetime.now().strftime('%Y_%m_%d_%H_%M')

    current_image = self.image_files[self.file_index]
    img_group = h5_file.create_group(f"{current_image.name}")
    img_group.create_dataset(
        "img", data=np.array(cv2.imread(str(current_image), 1))
    )
    contours_group = img_group.create_group("contours")

    n_contours = 0
    for item in range(len(self.contour_collection.items)):
        contour = self.contour_collection.items[item]
        if not contour.valid:
            continue
        if contour.navigation_window_contour is None:
            continue
        cv2_contours = contour.navigation_window_contour
        contours_group.create_dataset(f"cnt_{n_contours:06d}", data=np.array(cv2_contours))
        n_contours += 1

    h5_file.close()

image_width = 252
image_height = 1024
path = Path(os.getcwd())
create_random_dataset(image_width, image_height, 4, path)
