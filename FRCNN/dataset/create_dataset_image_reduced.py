import cv2
import h5py
from pathlib import Path
import random
import os
import numpy as np
from tqdm import tqdm

def write_contours(directory, file, data, header, mode="a"):
    with open(Path(directory, f"{file}"), mode, encoding="utf-8") as file:
        file.write(header + "\n")
        for line in data:
            file.write(str(line)[2:-2] + "\n")
        file.close()


def create_dataset(h5_path, output_path, N_VALIDATION, N_VERIFICATION):
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

    for h5_file in h5_files:
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

        for index, image_file in enumerate(image_files):

            if index in validation_indexes:
                CONTOURS_FOLDER = CONTOURS_FOLDER_VAL
                IMAGES_FOLDER = IMAGES_FOLDER_VAL
                DEBUG_FOLDER = DEBUG_FOLDER_VAL
            elif index in verification_indexes:
                CONTOURS_FOLDER = CONTOURS_FOLDER_VER
                IMAGES_FOLDER = IMAGES_FOLDER_VER
                DEBUG_FOLDER = DEBUG_FOLDER_VER
            else:
                CONTOURS_FOLDER = CONTOURS_FOLDER_TRAIN
                IMAGES_FOLDER = IMAGES_FOLDER_TRAIN
                DEBUG_FOLDER = DEBUG_FOLDER_TRAIN

            header = (
                "image_name, "
                "ellipse_center_x, ellipse_center_y, "
                "bbox_height, bbox_width, "
                "ellipse_main_axis, ellipse_secondary_axis, ellipse_angle"
                "\n"
            )

            output = []

            original_img = h5_dataset[image_file]["img"][...]
            marked_img = original_img.copy()

            for n_square in tqdm(range(1, 21), desc=f"Cropping image {index+1}/{N_images}", ncols=80, ascii=True, leave=True):

                base_filename = f"img_{Path(image_file).stem}"

                cnt_list_filename = f"{base_filename}_{n_square}_contours.txt"
                debug_image_filename = f"{base_filename}_{n_square}.jpg"
                raw_image_filename = f"{base_filename}_{n_square}.jpg"

                if Path(CONTOURS_FOLDER, cnt_list_filename).exists():
                    os.remove(Path(CONTOURS_FOLDER, cnt_list_filename))

                img_width = original_img.shape[1]
                img_height = original_img.shape[0]
                square_size = (img_width // 2, img_width // 2)

                x_center_random = random.randint(square_size[0] // 2, img_width - square_size[0] // 2)
                y_center_random = random.randint(square_size[0] // 2, img_height - square_size[0] // 2)

                x0_crop = x_center_random - square_size[0] // 2
                x1_crop = x_center_random + square_size[0] // 2
                y0_crop = y_center_random - square_size[0] // 2
                y1_crop = y_center_random + square_size[0] // 2

                original_img_cropped = original_img[y0_crop:y1_crop,
                                                    x0_crop:x1_crop]

                original_img_cropped = np.array(original_img_cropped, dtype=np.uint8)
                marked_img_original = original_img.copy()

                for contour_id in h5_dataset[image_file]["contours"]:
                    contour = h5_dataset[image_file]["contours"][contour_id]

                    contour_inside_region = True

                    for point in contour:
                        x, y = point[0]
                        if not (x0_crop <= x <= x1_crop and
                                y0_crop <= y <= y1_crop):
                            contour_inside_region = False

                    if contour_inside_region:
                        if len(contour[...]) < MIN_CONTOUR_LENGTH:
                            break

                        (
                            (center_x, center_y),
                            (ellipse_width, ellipse_height),
                            ellipse_angle,
                        ) = cv2.fitEllipse(contour[...])

                        center_x_translated = center_x - x0_crop
                        center_y_translated = center_y - y0_crop

                        marked_img_original = cv2.ellipse(
                            marked_img_original,
                            (int(center_x), int(center_y)),
                            (int(ellipse_width / 2), int(ellipse_height / 2)),
                            ellipse_angle,
                            0,
                            360,
                            (0, 0, 255),
                            2,
                        )

                        bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour[...])

                        bbox_x_translated = bbox_x - x0_crop
                        bbox_y_translated = bbox_y - y0_crop

                        marked_img_original = cv2.rectangle(
                            marked_img_original,
                            (bbox_x, bbox_y),
                            (bbox_x + bbox_width, bbox_y + bbox_height),
                            (0, 255, 0),
                            1,
                        )

                        output.append(
                            [
                                f"{raw_image_filename}, "
                                f"{int(center_x_translated):d}, {int(center_y_translated):d}, "
                                f"{int(bbox_height):d}, {int(bbox_width):d}, "
                                f"{int(ellipse_width):d}, {int(ellipse_height):d}, {ellipse_angle:.2f}"
                                f"\n"
                            ]
                        )

                marked_img_cropped = marked_img_original[y0_crop:y1_crop,
                                                         x0_crop:x1_crop]

                cv2.imwrite(str(Path(DEBUG_FOLDER, f"{debug_image_filename}")), marked_img_cropped)
                cv2.imwrite(str(Path(IMAGES_FOLDER, f"{raw_image_filename}")), original_img_cropped)

                with open(
                    Path(CONTOURS_FOLDER, f"{cnt_list_filename}"), "a", encoding="utf-8"
                ) as file:
                    file.write(header)
                    for line in output:
                        file.write("".join(line))
                    file.close()


create_dataset(os.getcwd(), os.getcwd(), 1, 1)
