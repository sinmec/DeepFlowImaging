import os
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Size of the sub-images
sub_image_size = 128

# Random mode option
random_samples = 64

# Number of Validation images
N_VALIDATION = 2

# Number of Verification images
N_VERIFICATION = 2


def create_dataset(path, sub_image_size, random_samples, N_VALIDATION, N_VERIFICATION):

    # Dataset folder
    dataset_folder = path

    IMAGES_FOLDER = Path(dataset_folder, "imgs_full")
    MASKS_FOLDER = Path(dataset_folder, "masks_full")

    FULL_IMAGES_FOLDER_TRAIN = Path(dataset_folder, "Training", "imgs_full")
    FULL_IMAGES_FOLDER_VAL = Path(dataset_folder, "Validation", "imgs_full")
    FULL_IMAGES_FOLDER_VER = Path(dataset_folder, "Verification", "imgs_full")

    FULL_MASKS_FOLDER_TRAIN = Path(dataset_folder, "Training", "masks_full")
    FULL_MASKS_FOLDER_VAL = Path(dataset_folder, "Validation", "masks_full")
    FULL_MASKS_FOLDER_VER = Path(dataset_folder, "Verification", "masks_full")

    SUB_IMAGES_FOLDER_TRAIN = Path(dataset_folder, "Training", "images")
    SUB_IMAGES_FOLDER_VAL = Path(dataset_folder, "Validation", "images")
    SUB_IMAGES_FOLDER_VER = Path(dataset_folder, "Verification", "images")

    SUB_MASKS_FOLDER_TRAIN = Path(dataset_folder, "Training", "masks")
    SUB_MASKS_FOLDER_VAL = Path(dataset_folder, "Validation", "masks")
    SUB_MASKS_FOLDER_VER = Path(dataset_folder, "Verification", "masks")

    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    MASKS_FOLDER.mkdir(parents=True, exist_ok=True)
    SUB_IMAGES_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    SUB_IMAGES_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    SUB_IMAGES_FOLDER_VER.mkdir(parents=True, exist_ok=True)
    SUB_MASKS_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    SUB_MASKS_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    SUB_MASKS_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    FULL_IMAGES_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    FULL_IMAGES_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    FULL_IMAGES_FOLDER_VER.mkdir(parents=True, exist_ok=True)
    FULL_MASKS_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    FULL_MASKS_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    FULL_MASKS_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    imgs = os.listdir(IMAGES_FOLDER)
    masks = os.listdir(MASKS_FOLDER)
    imgs.sort()
    masks.sort()

    N_images = len(imgs)
    shuffled_index = [*range(N_images)]
    random.Random(13).shuffle(shuffled_index)
    validation_indexes = shuffled_index[:N_VALIDATION]
    verification_indexes = shuffled_index[N_VALIDATION : N_VALIDATION + N_VERIFICATION]

    for index_i, img_file in tqdm(
        enumerate(imgs), total=N_images, desc="Creating sub-images"
    ):

        if index_i in validation_indexes:
            SUB_IMAGES_FOLDER = SUB_IMAGES_FOLDER_VAL
            SUB_MASKS_FOLDER = SUB_MASKS_FOLDER_VAL
            FULL_IMAGES_FOLDER = FULL_IMAGES_FOLDER_VAL
            FULL_MASKS_FOLDER = FULL_MASKS_FOLDER_VAL
        elif index_i in verification_indexes:
            SUB_IMAGES_FOLDER = SUB_IMAGES_FOLDER_VER
            SUB_MASKS_FOLDER = SUB_MASKS_FOLDER_VER
            FULL_IMAGES_FOLDER = FULL_IMAGES_FOLDER_VER
            FULL_MASKS_FOLDER = FULL_MASKS_FOLDER_VER
        else:
            SUB_IMAGES_FOLDER = SUB_IMAGES_FOLDER_TRAIN
            SUB_MASKS_FOLDER = SUB_MASKS_FOLDER_TRAIN
            FULL_IMAGES_FOLDER = FULL_IMAGES_FOLDER_TRAIN
            FULL_MASKS_FOLDER = FULL_MASKS_FOLDER_TRAIN

        img_base_name = Path(img_file).stem
        mask_file = f"{img_base_name}.png"
        img = cv2.imread(str(Path(IMAGES_FOLDER, img_file)), 0)
        img_mask = cv2.imread(str(Path(MASKS_FOLDER, mask_file)), 0)

        cv2.imwrite(str(Path(FULL_IMAGES_FOLDER, img_file)), img)
        cv2.imwrite(str(Path(FULL_MASKS_FOLDER, mask_file)), img_mask)

        random_cnt = 0
        for n in range(random_samples):
            rand_i = np.random.randint(low=0, high=img.shape[0] - sub_image_size)
            rand_j = np.random.randint(low=0, high=img.shape[1] - sub_image_size)

            i_start = rand_i
            i_end = rand_i + sub_image_size
            j_start = rand_j
            j_end = rand_j + sub_image_size
            img_split = img[i_start:i_end, j_start:j_end]
            mask_split = img_mask[i_start:i_end, j_start:j_end]

            cv2.imwrite(
                str(Path(SUB_IMAGES_FOLDER, f"{img_base_name}_{n:03d}.jpg")), img_split
            )
            cv2.imwrite(
                str(Path(SUB_MASKS_FOLDER, f"{img_base_name}_{n:03d}.png")), mask_split
            )

            random_cnt += 1
