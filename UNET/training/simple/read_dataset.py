import os
from pathlib import Path

import cv2
import numpy as np


def read_dataset(dataset_folder, window_size=256, subset="Training", debug=False):
    """
    This function reade the dataset from a dataset_folder.
    """

    images_folder = Path(dataset_folder, subset, "images")
    masks_folder = Path(dataset_folder, subset, "masks")

    all_images = os.listdir(images_folder)
    all_masks = os.listdir(masks_folder)

    N_images = len(all_images)
    N_masks = len(all_masks)

    images = np.zeros((N_images, window_size, window_size, 1), dtype=np.uint8)
    masks = np.zeros((N_masks, window_size, window_size, 1), dtype=np.uint8)

    for i in range(N_images):
        img_base_name = Path(all_images[i]).stem

        img_image = cv2.imread(str(Path(images_folder, "%s.jpg" % img_base_name)), 0)
        img_mask = cv2.imread(str(Path(masks_folder, "%s.png" % img_base_name)), 0)

        img_image = cv2.resize(img_image, (window_size, window_size))
        img_mask = cv2.resize(img_mask, (window_size, window_size))

        if debug:
            cv2.namedWindow("tst", cv2.WINDOW_NORMAL)
            cv2.imshow("tst", np.hstack((img_image, img_mask)))
            cv2.waitKey(0)

        images[i, :, :, 0] = img_image
        masks[i, :, :, 0] = img_mask

    images = images.astype(float)
    masks = masks.astype(float)
    images /= 255.0
    masks /= 255.0

    return images, masks
