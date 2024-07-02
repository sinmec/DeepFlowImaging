import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import cv2
import functools

def read_dataset(img_size, folder):

    # Defining the folders
    pwd = Path(folder)
    img_RAW_dir  = Path(pwd, 'images_raw')
    img_UNET_dir = Path(pwd, 'masks')
    label_dir    = Path(pwd, 'images_raw')

    # Listing the files inside the folder
    _imgs = os.listdir(img_UNET_dir)

    # Retrieving the png (image) files
    N_imgs = 0
    _imgs_png = []
    for img in _imgs:
        if img.endswith(".jpg"):
            N_imgs+=1
            _imgs_png.append(img.split(".jpg")[0])
    _imgs_png.sort()

    # Reading the csv and image files
    bbox_datasets= []
    imgs = np.zeros((N_imgs, img_size[0], img_size[1], 2),dtype=float)
    for i, _img in enumerate(_imgs_png):
        _file = _img

        # Reading the images (raw and mask)
        img_file_UNET = _file + ".jpg"
        img_file_RAW  = _file + ".jpg"
        img_UNET = cv2.imread(os.path.join(img_UNET_dir, img_file_UNET),0)
        img_RAW  = cv2.imread(os.path.join(img_RAW_dir, img_file_RAW),0)

        imgs[i,:,:,0] = img_UNET / 255.0
        imgs[i,:,:,1] = img_RAW / 255.0

        # Reading the corresponding csv file
        label_file = _file + ".csv"
        df = pd.read_csv(os.path.join(label_dir, label_file))
        df_np = df.to_numpy()
        N_circles = df_np.shape[0]

        # Multiplying by 1.1 to add an "extra room"
        df_np = df_np.astype(np.float64)
        bbox_dataset = df_np
        bbox_datasets.append(bbox_dataset)

    return imgs, bbox_datasets
