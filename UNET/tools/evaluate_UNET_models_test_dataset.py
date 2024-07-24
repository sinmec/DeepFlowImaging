import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from UNET.training.simple.read_dataset import read_dataset


models_dir = "../training/keras-tuner"

dataset_folder = Path(
    "/home/rafaelfc/Data/DeepFlowImaging/UNET/examples/dataset_UNET_PIV_IJMF/"
)

IMG_SIZE_RESHAPE = 128

keras_model_files = []
for keras_model_file in os.listdir(models_dir):
    if not keras_model_file.endswith(".keras"):
        continue
    keras_model_files.append(Path(keras_model_file))

for keras_model_file in keras_model_files:

    model_name = keras_model_file.stem

    model_folder_name = Path(model_name, "test_dataset")
    model_folder_name.mkdir(exist_ok=True, parents=True)

    images_verification, masks_verification = read_dataset(
        dataset_folder, window_size=IMG_SIZE_RESHAPE, subset="Verification", debug=False
    )

    UNET_model = keras.models.load_model(
        Path(models_dir, keras_model_file), compile=True
    )

    test_loss = UNET_model.evaluate(images_verification, masks_verification)
    print(f"OUT: {model_name}, {test_loss:.3e}")

    UNET_imgs = UNET_model.predict(images_verification)

    UNET_imgs *= 255.0
    UNET_imgs = UNET_imgs.astype(np.uint8)
    raw_imgs = images_verification.copy()
    raw_imgs *= 255.0
    raw_imgs = raw_imgs.astype(np.uint8)
    mask_imgs = masks_verification.copy()
    mask_imgs *= 255.0
    mask_imgs = mask_imgs.astype(np.uint8)
    N_imgs = UNET_imgs.shape[0]
    for k in range(N_imgs):
        raw_img = raw_imgs[k, :, :, 0]
        mask_img = mask_imgs[k, :, :, 0]
        UNET_img = UNET_imgs[k, :, :, 0]
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        UNET_img = cv2.cvtColor(UNET_img, cv2.COLOR_GRAY2BGR)

        debug_img = raw_img.copy()
        debug_img[UNET_img[:, :, 2] > 120, 2] = 255
        img_out = np.hstack((raw_img, mask_img, UNET_img, debug_img))
        img_out_filename = Path(model_folder_name, f"img_{k:06d}.jpg")
        cv2.imwrite(str(img_out_filename), img_out)
