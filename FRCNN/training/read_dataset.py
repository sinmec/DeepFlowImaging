import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import cv2
import functools

def read_dataset(img_size, dataset_folder, subset):
    pwd = Path(dataset_folder)

    images_folder = Path(pwd, subset, "images")
    masks_folder = Path(pwd, subset, "debug")
    txt_folder = Path(pwd, subset, "contours")

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
    imgs = np.zeros((N_imgs, img_size[0], img_size[1], 2), dtype=float)

    for i, _img in enumerate(_imgs_png):
        _file = _img

        # Reading the images (raw and mask)
        mask_img_file = f"{_file}.jpg"
        raw_img_file = f"{_file}.jpg"
        mask_img = cv2.imread(str(Path(masks_folder, mask_img_file)), 0)
        raw_img = cv2.imread(str(Path(images_folder, raw_img_file)), 0)

        imgs[i, :, :, 0] = mask_img / 255.0
        imgs[i, :, :, 1] = raw_img / 255.0

        # Reading the corresponding .txt file
        label_file = f"{_file}_contours.txt"
        df = pd.read_csv(str(Path(txt_folder, label_file)), sep=", ", usecols=[1, 2, 3, 4, 5, 6, 7], engine='python')
        df_np = df.to_numpy()
        N_circles = df_np.shape[0]

        df_np = df_np.astype(np.float64)
        bbox_dataset = df_np
        bbox_datasets.append(bbox_dataset)



    return imgs, bbox_datasets
