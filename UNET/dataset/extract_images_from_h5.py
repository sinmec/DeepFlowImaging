import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def extract_images_from_h5(h5_dataset_path, output_path, contour_line_width=cv2.FILLED):
    h5_dataset_folder = Path(h5_dataset_path)
    output_folder = Path(output_path)

    h5_dataset_folder.mkdir(exist_ok=True, parents=True)
    output_folder.mkdir(exist_ok=True, parents=True)

    h5_files = []
    for h5_file in os.listdir(h5_dataset_folder):
        if ".h5" not in h5_file:
            continue
        h5_files.append(h5_file)

    print(h5_files)
    for h5_file in tqdm(
            h5_files, total=len(h5_files), desc="Extracting dataset from .h5 file"
    ):

        imgs_full_dir = Path(output_folder, "imgs_full")
        masks_full_dir = Path(output_folder, "masks_full")

        masks_full_dir.mkdir(exist_ok=True, parents=True)
        imgs_full_dir.mkdir(exist_ok=True, parents=True)

        h5_dataset = h5py.File(f"{h5_dataset_folder / h5_file}", "r")
        image_files = h5_dataset.keys()

        for image_file in image_files:

            original_img = h5_dataset[image_file]["img"][...]
            mask_img = np.zeros(original_img.shape, dtype=np.uint8)
            output_image_file = Path(image_file).stem

            for contour_id in h5_dataset[image_file]["contours"]:
                contour = h5_dataset[image_file]["contours"][contour_id]
                cv2.drawContours(
                    mask_img, [contour[...]], -1, [255, 255, 255], contour_line_width
                )

            cv2.imwrite(
                str(Path(imgs_full_dir, f"{output_image_file}.jpg")), original_img
            )
            cv2.imwrite(str(Path(masks_full_dir, f"{output_image_file}.png")), mask_img)
