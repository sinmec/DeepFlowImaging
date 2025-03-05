import cv2
import numpy as np

import config as cfg
from FRCNN.training.simple.return_bounding_box_points import return_bounding_box_points


def create_samples_for_training(
    anchors,
    labels,
    bbox_dataset,
    image,
    debug=False,
):

    if debug:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        image = image.copy() * 255.0
        image_rgb = image.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

    n_pos = int(cfg.POS_RATIO * cfg.N_SAMPLES)

    pos_index = np.where(labels == 1)[0]

    if len(pos_index) > n_pos:
        disable_index = np.random.choice(
            pos_index, size=(len(pos_index) - n_pos), replace=False
        )
        labels[disable_index] = -1

    n_neg = cfg.N_SAMPLES - np.sum(labels == 1)

    if cfg.SHOW_N_POS:
        print(
            "n_posi = %d / %d(set)"
            % (np.sum(labels == 1), cfg.POS_RATIO * cfg.N_SAMPLES)
        )

    neg_index = np.where(labels == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(
            neg_index, size=(len(neg_index) - n_neg), replace=False
        )
        labels[disable_index] = -1

    if debug:
        for k, bbox in enumerate(bbox_dataset):
            # x_b_1 = int(bbox[0] - (bbox[3] / 2))
            # y_b_1 = int(bbox[1] - (bbox[2] / 2))
            # x_b_2 = int(bbox[0] + (bbox[3] / 2))
            # y_b_2 = int(bbox[1] + (bbox[2] / 2))
            p_1, p_2 = return_bounding_box_points(bbox)
            cv2.rectangle(image_rgb, p_1, p_2, (0, 255, 255), 4)

        for k in range(len(labels)):
            if labels[k] == 1:
                anchor = anchors[k]
                x_a_1 = int(anchor[0])
                y_a_1 = int(anchor[1])
                x_a_2 = int(anchor[2])
                y_a_2 = int(anchor[3])
                p_1 = (x_a_1, y_a_1)
                p_2 = (x_a_2, y_a_2)
                cv2.rectangle(image_rgb, p_1, p_2, (255, 255, 0), 4)

        cv2.imshow("test", image_rgb)
        cv2.waitKey(0)

    return labels
