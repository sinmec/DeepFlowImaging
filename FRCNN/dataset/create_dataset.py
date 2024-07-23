import sys

import cv2
import h5py
from pathlib import Path
import random
import os

import numpy as np
from tensorflow import keras
from tqdm import tqdm

sys.path.append('../../UNET/')
from prediction.apply_UNET_mask_split import apply_UNET_mask
from prediction.divide_image import divide_image
from prediction.recreate_UNET_image import recreate_UNET_image

def write_contours(directory, file, data, header, mode="a"):
    with open(Path(directory, f"{file}"), mode, encoding="utf-8") as file:
        file.write(header + "\n")
        for line in data:
            file.write(str(line)[2:-2] + "\n")
        file.close()


def create_dataset(h5_path, output_path, N_VALIDATION, N_VERIFICATION, UNET_model_options=None):

    if UNET_model_options:
        UNET_model = keras.models.load_model(UNET_model_options['keras_file'], compile=False)
        window_size = UNET_model_options['window_size']
        sub_image_size = UNET_model_options['sub_image_size']
        stride_division = UNET_model_options['stride_division']


    dataset_path = Path(output_path, "Output")

    CONTOURS_FOLDER_TRAIN = Path(dataset_path, "Training", "contours")
    CONTOURS_FOLDER_VAL = Path(dataset_path, "Validation", "contours")
    CONTOURS_FOLDER_VER = Path(dataset_path, "Verification", "contours")

    DEBUG_FOLDER_TRAIN = Path(dataset_path, "Training", "debug")
    DEBUG_FOLDER_VAL = Path(dataset_path, "Validation", "debug")
    DEBUG_FOLDER_VER = Path(dataset_path, "Verification", "debug")

    IMAGES_FOLDER_TRAIN = Path(dataset_path, "Training", "images")
    IMAGES_FOLDER_VAL = Path(dataset_path, "Validation", "images")
    IMAGES_FOLDER_VER = Path(dataset_path, "Verification", "images")

    if UNET_model_options:
        MASKS_FOLDER_TRAIN = Path(dataset_path, "Training", "masks")
        MASKS_FOLDER_VAL = Path(dataset_path, "Validation", "masks")
        MASKS_FOLDER_VER = Path(dataset_path, "Verification", "masks")

        MASKS_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
        MASKS_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
        MASKS_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    CONTOURS_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    CONTOURS_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    CONTOURS_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    DEBUG_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    DEBUG_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    DEBUG_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    IMAGES_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    IMAGES_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    IMAGES_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    h5_files = []
    for h5_file in os.listdir(h5_path):
        if ".h5" not in h5_file:
            continue
        h5_files.append(h5_file)

    for h5_file in tqdm(
            h5_files, total=len(h5_files), desc="Extracting dataset from .h5 file"
    ):
        h5_dataset = h5py.File(Path(h5_path, h5_file), "r")
        image_files = h5_dataset.keys()

        N_images = len(image_files)
        shuffled_index = [*range(N_images)]
        random.Random(13).shuffle(shuffled_index)
        validation_indexes = shuffled_index[:N_VALIDATION]
        verification_indexes = shuffled_index[
            N_VALIDATION : N_VALIDATION + N_VERIFICATION
        ]

        MIN_CONTOUR_LENGTH = 5

        for index, image_file in tqdm(
                enumerate(image_files), total=len(image_files), desc="Creating dataset"
        ):

            if index in validation_indexes:
                CONTOURS_FOLDER = CONTOURS_FOLDER_VAL
                IMAGES_FOLDER = IMAGES_FOLDER_VAL
                DEBUG_FOLDER = DEBUG_FOLDER_VAL
                if UNET_model_options:
                    MASKS_FOLDER = MASKS_FOLDER_VAL
            elif index in verification_indexes:
                CONTOURS_FOLDER = CONTOURS_FOLDER_VER
                IMAGES_FOLDER = IMAGES_FOLDER_VER
                DEBUG_FOLDER = DEBUG_FOLDER_VER
                if UNET_model_options:
                    MASKS_FOLDER = MASKS_FOLDER_VER
            else:
                CONTOURS_FOLDER = CONTOURS_FOLDER_TRAIN
                IMAGES_FOLDER = IMAGES_FOLDER_TRAIN
                DEBUG_FOLDER = DEBUG_FOLDER_TRAIN
                if UNET_model_options:
                    MASKS_FOLDER = MASKS_FOLDER_TRAIN

            header = (
                "image_name, "
                "ellipse_center_x, ellipse_center_y, "
                "bbox_height, bbox_width, "
                "ellipse_main_axis, ellipse_secondary_axis, ellipse_angle"
                "\n"
            )

            output = []

            base_filename = f"img_{Path(image_file).stem}"

            cnt_list_filename = f"{base_filename}_contours.txt"
            debug_image_filename = f"{base_filename}.jpg"
            raw_image_filename = f"{base_filename}.jpg"

            if Path(CONTOURS_FOLDER, cnt_list_filename).exists():
                os.remove(Path(CONTOURS_FOLDER, cnt_list_filename))

            original_img = h5_dataset[image_file]["img"][...]
            marked_img = original_img.copy()

            if UNET_model_options:
                subdivided_image_raw = divide_image(
                    window_size, sub_image_size, original_img[:,:,0], stride_division
                )

                subdivided_image_UNET = apply_UNET_mask(subdivided_image_raw, UNET_model)
                img_UNET = recreate_UNET_image(
                    subdivided_image_UNET, window_size, original_img[:,:,0], stride_division
                )
                marked_img_UNET = cv2.cvtColor(img_UNET.copy(), cv2.COLOR_GRAY2BGR)


            for contour_id in h5_dataset[image_file]["contours"]:
                contour = h5_dataset[image_file]["contours"][contour_id]

                if len(contour[...]) < MIN_CONTOUR_LENGTH:
                    break

                (
                    (center_x, center_y),
                    (ellipse_width, ellipse_height),
                    ellipse_angle,
                ) = cv2.fitEllipse(contour[...])

                marked_img = cv2.ellipse(
                    marked_img,
                    (int(center_x), int(center_y)),
                    (int(ellipse_width / 2), int(ellipse_height / 2)),
                    ellipse_angle,
                    0,
                    360,
                    (0, 0, 255),
                    2,
                )



                bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour[...])
                marked_img = cv2.rectangle(
                    marked_img,
                    (bbox_x, bbox_y),
                    (bbox_x + bbox_width, bbox_y + bbox_height),
                    (0, 255, 0),
                    1,
                )

                if UNET_model_options:
                    marked_img_UNET = cv2.ellipse(
                        marked_img_UNET,
                        (int(center_x), int(center_y)),
                        (int(ellipse_width / 2), int(ellipse_height / 2)),
                        ellipse_angle,
                        0,
                        360,
                        (0, 0, 255),
                        2,
                    )

                    marked_img_UNET = cv2.rectangle(
                        marked_img_UNET,
                        (bbox_x, bbox_y),
                        (bbox_x + bbox_width, bbox_y + bbox_height),
                        (0, 255, 0),
                        1,
                    )

                output.append(
                    [
                        f"{raw_image_filename}, "
                        f"{int(center_x):d}, {int(center_y):d}, "
                        f"{int(bbox_height):d}, {int(bbox_width):d}, "
                        f"{int(ellipse_width):d}, {int(ellipse_height):d}, {ellipse_angle:.2f}"
                        f"\n"
                    ]
                )

            cv2.imwrite(str(Path(DEBUG_FOLDER, f"{debug_image_filename}")), marked_img)
            cv2.imwrite(str(Path(IMAGES_FOLDER, f"{raw_image_filename}")), original_img)

            if UNET_model_options:
                cv2.imwrite(str(Path(MASKS_FOLDER, f"{raw_image_filename}")), img_UNET)
                cv2.imwrite(str(Path(DEBUG_FOLDER, f"{debug_image_filename}")), np.hstack((marked_img, marked_img_UNET)))

            with open(
                Path(CONTOURS_FOLDER, f"{cnt_list_filename}"), "a", encoding="utf-8"
            ) as file:
                file.write(header)
                for line in output:
                    file.write("".join(line))
                file.close()
