import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Path to .h5 file - Modify it accordingly
directory = Path(r"/EXPERIMENTS/DeepFlowImaging/UNET_examples/h5_multilabeller_files")

# Dataset path - output folder
dataset_dir = Path("dataset_UNET")

h5_files = []
for h5_file in os.listdir(directory):
    if ".h5" not in h5_file:
        continue
    h5_files.append(h5_file)

for h5_files in tqdm(h5_files, total=len(h5_files), desc="Extracting dataset from .h5 file"):

    imgs_full_dir = Path(dataset_dir, "imgs_full")
    masks_full_dir = Path(dataset_dir, "masks_full")

    masks_full_dir.mkdir(exist_ok=True, parents=True)
    imgs_full_dir.mkdir(exist_ok=True, parents=True)

    h5_dataset = h5py.File(f"{directory / h5_file}", "r")
    image_files = h5_dataset.keys()

    for image_file in image_files:

        original_img = h5_dataset[image_file]["img"][...]
        mask_img = np.zeros(original_img.shape, dtype=np.uint8)

        for contour_id in h5_dataset[image_file]["contours"]:
            contour = h5_dataset[image_file]["contours"][contour_id]
            cv2.drawContours(mask_img, [contour[...]], -1, [255, 255, 255], cv2.FILLED)

        cv2.imwrite(str(Path(imgs_full_dir, f"img_{image_file[:4]}.jpg")), original_img)
        cv2.imwrite(str(Path(masks_full_dir, f"img_{image_file[:4]}.png")), mask_img)
