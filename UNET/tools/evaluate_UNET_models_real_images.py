import os
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from UNET.prediction.apply_UNET_mask_split import apply_UNET_mask
from UNET.prediction.divide_image import divide_image
from UNET.prediction.recreate_UNET_image import recreate_UNET_image

models_dir = "../training/keras-tuner"

# Folder where the test images are stored
raw_images_dir = Path("/home/rafaelfc/Data/DeepFlowImaging/UNET/examples/dataset_UNET_PIV_IJMF/Verification/imgs_full")
mask_images_dir = Path(
    "/home/rafaelfc/Data/DeepFlowImaging/UNET/examples/dataset_UNET_PIV_IJMF/Verification/masks_full")

# Defining the UNET image properties
STRIDE_DIVISIONS = 16
WINDOW_SIZE = 512
WINDOW_SIZE_RESHAPE = 128

keras_model_files = []
for keras_model_file in os.listdir(models_dir):
    if not keras_model_file.endswith(".keras"):
        continue
    keras_model_files.append(Path(keras_model_file))

for i, keras_model_file in enumerate(tqdm(keras_model_files, desc="Evaluating models")):

    model_name = keras_model_file.stem

    model_folder_name = Path(model_name, "UNET_recreate")
    model_folder_name.mkdir(exist_ok=True, parents=True)

    test_raw_files = os.listdir(raw_images_dir)
    test_masks_files = os.listdir(mask_images_dir)

    test_raw_files.sort()
    test_masks_files.sort()

    test_masks_imgs = []
    test_raw_imgs = []
    for img in test_raw_files:
        _img = cv2.imread(os.path.join(raw_images_dir, img), 0)
        test_raw_imgs.append(_img)
    for img in test_masks_files:
        _img = cv2.imread(os.path.join(mask_images_dir, img), 0)
        test_masks_imgs.append(_img)

    UNET_model = keras.models.load_model(os.path.join(models_dir, keras_model_file), compile=False)

    N_test = len(test_raw_imgs)

    RMS_value = 0.0
    for index, img in enumerate(test_raw_imgs):
        print("Img %06d / %06d" % (index, N_test))

        test_mask = test_masks_imgs[index]

        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # cv2.imshow("test", test_mask)
        # cv2.waitKey(0)

        img_shape = img.shape
        N_pixels = img.shape[0] * img.shape[1]

        image_subdivided = divide_image(WINDOW_SIZE,
                                        WINDOW_SIZE_RESHAPE,
                                        img,
                                        STRIDE_DIVISIONS)

        sub_images_UNET = apply_UNET_mask(image_subdivided, UNET_model)

        UNET_img = recreate_UNET_image(sub_images_UNET, WINDOW_SIZE, img, STRIDE_DIVISIONS)
        UNET_orig_img_uint8 = UNET_img.copy()

        UNET_img[UNET_img >= 122] = 255
        UNET_img[UNET_img < 122] = 0

        raw_img_uint8 = img.copy()
        test_img_uint8 = test_mask.copy()

        UNET_img_uint8 = UNET_img.copy()

        img = img.astype(float)
        test_mask = test_mask.astype(float)
        UNET_img = UNET_img.astype(float)

        sqr_difference = np.power((UNET_img - test_mask), 2.0)
        pix_sum = np.sum(sqr_difference)
        RMS_value += pix_sum

        img_diff = np.abs(test_mask - UNET_img)
        img_diff = img_diff.astype(np.uint8)

        img_out = np.hstack((raw_img_uint8, test_img_uint8, UNET_orig_img_uint8, UNET_img_uint8, img_diff))
        cv2.imwrite(os.path.join(model_folder_name, "img_%06d.jpg" % index), img_out)
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # cv2.imshow("test", img_diff)
        # cv2.waitKey(0)

    RMS_value /= (N_test * N_pixels)
    RMS_value = np.sqrt(RMS_value)

    print(f'OUT: {model_name}, {RMS_value:.3e}')
