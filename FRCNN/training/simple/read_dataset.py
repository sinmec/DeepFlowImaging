import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_dataset(img_size, dataset_folder, subset="Training"):

    images_folder = Path(dataset_folder, subset, "images")
    masks_folder = Path(dataset_folder, subset, "masks")
    txt_folder = Path(dataset_folder, subset, "contours")

    _imgs = os.listdir(masks_folder)

    N_imgs = 0
    _imgs_png = []
    for img in _imgs:
        if img.endswith(".jpg"):
            N_imgs += 1
            _imgs_png.append(img.split(".jpg")[0])
    _imgs_png.sort()

    # Reading the csv and image files
    bbox_datasets = []
    imgs = np.zeros(
        (
            N_imgs,
            img_size[0],
            img_size[1],
        ),
        dtype=float,
    )
    raw_imgs = np.zeros(
        (
            N_imgs,
            img_size[0],
            img_size[1],
        ),
        dtype=float,
    )

    for i, _img in enumerate(_imgs_png):
        _file = _img

        # Reading the images (raw and mask)
        mask_img_file = f"{_file}.jpg"
        raw_img_file = f"{_file}.jpg"
        mask_img = cv2.imread(str(Path(masks_folder, mask_img_file)), 0)
        raw_img = cv2.imread(str(Path(images_folder, raw_img_file)), 0)

        imgs[i, :, :] = mask_img / 255.0
        raw_imgs[i, :, :] = raw_img / 255.0

        # Reading the corresponding .txt file
        label_file = f"{_file}_contours.txt"
        cols_to_use = [
            "ellipse_center_x",
            "ellipse_center_y",
            "bbox_height",
            "bbox_width",
        ]
        df = pd.read_csv(
            str(Path(txt_folder, label_file)),
            sep=", ",
            usecols=cols_to_use,
            engine="python",
            index_col=False,
        )[cols_to_use]
        df_np = df.to_numpy()

        df_np = df_np.astype(np.float64)
        bbox_dataset = df_np
        bbox_datasets.append(bbox_dataset)

    debug = False
    if debug:
        i = 0
        for k in range(raw_imgs.shape[0]):
            __img = raw_imgs[k] * 255.0
            __img = __img.astype(np.uint8)
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2BGR)

            __bbox_dataset = bbox_datasets[k]
            for __bbox in __bbox_dataset:
                x_b_1 = int(__bbox[0] - (__bbox[2] / 2))
                y_b_1 = int(__bbox[1] - (__bbox[3] / 2))
                x_b_2 = int(__bbox[0] + (__bbox[2] / 2))
                y_b_2 = int(__bbox[1] + (__bbox[3] / 2))
                c_x = (x_b_1 + x_b_2) // 2
                c_y = (y_b_1 + y_b_2) // 2
                p_1 = (x_b_1, y_b_1)
                p_2 = (x_b_2, y_b_2)
                cv2.rectangle(__img, p_1, p_2, (0, 255, 255), 4)

            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.imshow("test", __img)
            cv2.waitKey(0)
        i += 1

    return imgs, bbox_datasets, raw_imgs
