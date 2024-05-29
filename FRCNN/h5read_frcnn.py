import cv2
import h5py
from pathlib import Path


def generate_frcnn_dataset(h5_path):
    # Reading the .h5 file
    h5_dataset = h5py.File(h5_path, 'r')

    # Get images
    image_files = h5_dataset.keys()

    # Creating output directories
    Path('output').mkdir(parents=True, exist_ok=True)
    Path('output/contours_info').mkdir(parents=True, exist_ok=True)
    Path('output/train').mkdir(parents=True, exist_ok=True)
    Path('output/validation').mkdir(parents=True, exist_ok=True)

    for image_file in image_files:

        header = "image_name, ellipse_center_x, ellipse_center_y, bbox_height, bbox_width," \
                 " ellipse_main_axis, ellipse_secondary_axis, ellipse_angle"

        output = []

        file_name = f'img_{image_file[0:-4]}_contours.txt'

        original_img = h5_dataset[image_file]['img'][...]
        marked_img = original_img.copy()

        for contour_id in h5_dataset[image_file]['contours']:
            contour = h5_dataset[image_file]['contours'][contour_id]

            if len(contour[...]) >= 5:
                ((center_x, center_y), (ellipse_width, ellipse_height), ellipse_angle) = cv2.fitEllipse(contour[...])

                marked_img = cv2.ellipse(marked_img, (int(center_x), int(center_y)),
                                         (int(ellipse_width / 2), int(ellipse_height / 2)),
                                         ellipse_angle, 0, 360, (0, 0, 255), 2)
            else:
                break

            bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour[...])
            marked_img = cv2.rectangle(marked_img, (bbox_x, bbox_y),
                                       (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0), 1)

            output.append([f"{image_file}, {center_x:.2f}, {center_y:.2f}, {bbox_height:.2f}, {bbox_width:.2f},"
                           f" {ellipse_width:.2f}, {ellipse_height:.2f}, {ellipse_angle:.2f}"])

        cv2.imwrite(f'output/train/cnts_{image_file}', marked_img)
        cv2.imwrite(f'output/validation/cnts_{image_file}', original_img)

        with open(f'output/contours_info/{file_name}', 'w', encoding='utf-8') as file:
            file.write(header + '\n')
            for linha in output:
                file.write(str(linha)[2:-2] + '\n')


h5_file = Path(r"data/data.h5")

generate_frcnn_dataset(h5_file)
