import cv2
import h5py
from pathlib import Path
import random
import os


def create_dataset(h5_path, output_path, N_VALIDATION, N_VERIFICATION):
    dataset_path = Path(output_path, "Output")

    TRAINING_FOLDER = Path(dataset_path, "Training")
    VALIDATION_FOLDER = Path(dataset_path, "Validation")
    VERIFICATION_FOLDER = Path(dataset_path, "Verification")

    SUB_CONTOURS_FOLDER_TRAIN = Path(dataset_path, "Training", "contours")
    SUB_CONTOURS_FOLDER_VAL = Path(dataset_path, "Validation", "contours")
    SUB_CONTOURS_FOLDER_VER = Path(dataset_path, "Verification", "contours")

    SUB_DEBUG_FOLDER_TRAIN = Path(dataset_path, "Training", "debug")
    SUB_DEBUG_FOLDER_VAL = Path(dataset_path, "Validation", "debug")
    SUB_DEBUG_FOLDER_VER = Path(dataset_path, "Verification", "debug")

    SUB_IMAGES_FOLDER_TRAIN = Path(dataset_path, "Training", "images")
    SUB_IMAGES_FOLDER_VAL = Path(dataset_path, "Validation", "images")
    SUB_IMAGES_FOLDER_VER = Path(dataset_path, "Verification", "images")

    TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)
    VALIDATION_FOLDER.mkdir(parents=True, exist_ok=True)
    VERIFICATION_FOLDER.mkdir(parents=True, exist_ok=True)
    SUB_CONTOURS_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    SUB_CONTOURS_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    SUB_CONTOURS_FOLDER_VER.mkdir(parents=True, exist_ok=True)
    SUB_DEBUG_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    SUB_DEBUG_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    SUB_DEBUG_FOLDER_VER.mkdir(parents=True, exist_ok=True)
    SUB_IMAGES_FOLDER_TRAIN.mkdir(parents=True, exist_ok=True)
    SUB_IMAGES_FOLDER_VAL.mkdir(parents=True, exist_ok=True)
    SUB_IMAGES_FOLDER_VER.mkdir(parents=True, exist_ok=True)

    def write_contours(directory, file, data, header):
        with open(Path(directory, f'{file}'), 'w', encoding='utf-8') as file:
            file.write(header + '\n')
            for line in data:
                file.write(str(line)[2:-2] + '\n')

    h5_files = []
    for h5_file in os.listdir(h5_path):
        if ".h5" not in h5_file:
            continue
        h5_files.append(h5_file)

    for h5_file in h5_files:
        h5_dataset = h5py.File(Path(h5_path, h5_file), 'r')
        image_files = h5_dataset.keys()

        image_files_list = list(image_files)
        N_images = len(image_files)
        shuffled_index = [*range(N_images)]
        random.Random(13).shuffle(shuffled_index)
        validation_indexes = shuffled_index[:N_VALIDATION]
        verification_indexes = shuffled_index[N_VALIDATION: N_VALIDATION + N_VERIFICATION]

        MIN_CONTOUR_LENGHT = 5

        for image_file in image_files:

            header = ("image_name, "
                      "ellipse_center_x, ellipse_center_y, "
                      "bbox_height, bbox_width, "
                      "ellipse_main_axis, ellipse_secondary_axis, ellipse_angle")

            output = []

            base_filename = f'img_{Path(image_file).stem}'

            cnt_list_filename = f'{base_filename}_contours.txt'
            debug_image_filename = f'{base_filename}.jpg'
            raw_image_filename = f'{base_filename}.jpg'


            original_img = h5_dataset[image_file]['img'][...]
            marked_img = original_img.copy()

            for contour_id in h5_dataset[image_file]['contours']:
                contour = h5_dataset[image_file]['contours'][contour_id]

                if len(contour[...]) < MIN_CONTOUR_LENGHT:
                    break

                ((center_x, center_y), (ellipse_width, ellipse_height), ellipse_angle) = cv2.fitEllipse(contour[...])

                marked_img = cv2.ellipse(marked_img, (int(center_x), int(center_y)),
                                         (int(ellipse_width / 2), int(ellipse_height / 2)),
                                         ellipse_angle, 0, 360, (0, 0, 255), 2)

                bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour[...])
                marked_img = cv2.rectangle(marked_img, (bbox_x, bbox_y),
                                           (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0), 1)

                output.append([f"{raw_image_filename}, "
                               f"{int(center_x):d}, {int(center_y):d}, "
                               f"{int(bbox_height):d}, {int(bbox_width):d}, "
                               f"{int(ellipse_width):d}, {int(ellipse_height):d}, {ellipse_angle:.2f}"])

            SUB_DEBUG_FOLDERS = [SUB_DEBUG_FOLDER_VAL, SUB_DEBUG_FOLDER_VER, SUB_DEBUG_FOLDER_TRAIN]

            for folder in SUB_DEBUG_FOLDERS:
                image_files_list = list(image_files)
                index = image_files_list.index(image_file)
                if folder == SUB_DEBUG_FOLDER_VAL:
                    if index in validation_indexes:
                        cv2.imwrite(str(Path(folder, f'{debug_image_filename}')), marked_img)
                elif folder == SUB_DEBUG_FOLDER_VER:
                    if index in verification_indexes:
                        cv2.imwrite(str(Path(folder, f'{debug_image_filename}')), marked_img)
                else:
                    cv2.imwrite(str(Path(folder, f'{debug_image_filename}')), marked_img)

            SUB_IMAGES_FOLDERS = [SUB_IMAGES_FOLDER_TRAIN, SUB_IMAGES_FOLDER_VAL, SUB_IMAGES_FOLDER_VER]

            for folder in SUB_IMAGES_FOLDERS:
                index = image_files_list.index(image_file)
                if folder == SUB_IMAGES_FOLDER_VAL:
                    if index in validation_indexes:
                        cv2.imwrite(str(Path(folder, f'{raw_image_filename}')), original_img)
                elif folder == SUB_IMAGES_FOLDER_VER:
                    if index in verification_indexes:
                        cv2.imwrite(str(Path(folder, f'{raw_image_filename}')), original_img)
                else:
                    cv2.imwrite(str(Path(folder, f'{raw_image_filename}')), original_img)

            SUB_CONTOURS_FOLDERS = [SUB_CONTOURS_FOLDER_TRAIN, SUB_CONTOURS_FOLDER_VAL, SUB_CONTOURS_FOLDER_VER]

            for folder in SUB_CONTOURS_FOLDERS:
                index = image_files_list.index(image_file)
                if folder == SUB_CONTOURS_FOLDER_VAL:
                    if index in validation_indexes:
                        write_contours(folder, cnt_list_filename, output, header)
                elif folder == SUB_CONTOURS_FOLDER_VER:
                    if index in verification_indexes:
                        write_contours(folder, cnt_list_filename, output, header)
                else:
                    write_contours(folder, cnt_list_filename, output, header)
