import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from FRCNN.training.simple.return_bounding_box_points import return_bounding_box_points


def read_dataset(img_size, dataset_folder, mode="raw", subset="Training"):

    images_folder = Path(dataset_folder, subset, "images")
    txt_folder = Path(dataset_folder, subset, "contours")
    masks_folder = images_folder
    if mode == "mask":
        masks_folder = Path(dataset_folder, subset, "masks")

    _imgs = os.listdir(masks_folder)

    N_imgs = 0
    image_files = []
    for img in _imgs:
        if img.endswith(".jpg"):
            N_imgs += 1
            image_files.append(img.split(".jpg")[0])
    image_files.sort()

    mask_imgs = np.zeros(
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

    contour_files = []
    for i, _img in enumerate(image_files):
        _file = _img

        # Reading the images (raw and mask)
        mask_img_file = f"{_file}.jpg"
        raw_img_file = f"{_file}.jpg"
        mask_img = cv2.imread(str(Path(masks_folder, mask_img_file)), 0)
        raw_img = cv2.imread(str(Path(images_folder, raw_img_file)), 0)

        mask_imgs[i, :, :] = mask_img / 255.0
        raw_imgs[i, :, :] = raw_img / 255.0

        contour_files.append(f"{_file}_contours.txt")

    MAX_BBOXES_PER_IMAGE = 200
    bbox_datasets = np.zeros(
        (len(contour_files), MAX_BBOXES_PER_IMAGE, 4), dtype=np.float32
    )
    bbox_datasets[...] = np.nan

    for index_contour, contour_file in enumerate(contour_files):
        cols_to_use = [
            "ellipse_center_x",
            "ellipse_center_y",
            "bbox_height",
            "bbox_width",
        ]

        df = pd.read_csv(
            str(Path(txt_folder, contour_file)),
            sep=", ",
            usecols=cols_to_use,
            engine="python",
            index_col=False,
        )[cols_to_use]
        df_np = df.to_numpy()
        for index_bbox, _df_np in enumerate(df_np):
            assert (
                index_bbox <= MAX_BBOXES_PER_IMAGE
            ), "Exceeded maximum allowed bounding boxes per image!"

            bbox_datasets[index_contour, index_bbox, 0] = _df_np[0]
            bbox_datasets[index_contour, index_bbox, 1] = _df_np[1]
            bbox_datasets[index_contour, index_bbox, 2] = _df_np[2]
            bbox_datasets[index_contour, index_bbox, 3] = _df_np[3]

    debug = False
    if debug:
        for k in range(raw_imgs.shape[0]):
            __img = raw_imgs[k] * 255.0
            __img = __img.astype(np.uint8)
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2BGR)

            __bbox_dataset = bbox_datasets[k]
            for __bbox in __bbox_dataset:
                if np.isnan(__bbox[0]):
                    continue
                p_1, p_2 = return_bounding_box_points(__bbox)
                cv2.rectangle(__img, p_1, p_2, (0, 255, 255), 4)

            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.imshow("test", __img)
            cv2.waitKey(0)

    return mask_imgs, bbox_datasets, raw_imgs
