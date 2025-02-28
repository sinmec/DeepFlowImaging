import cv2
import numpy as np
import math


def divide_image(raw_img, window_size, stride_division=2):
    scan_I = np.arange(
        0, raw_img.shape[0] - window_size + 1, window_size // stride_division
    )
    scan_J = np.arange(
        0, raw_img.shape[1] - window_size + 1, window_size // stride_division
    )

    raw_img_subdivided = np.zeros(
        (scan_I.shape[0] * scan_J.shape[0], window_size, window_size),
        dtype=np.float64,
    )

    positions = np.zeros((scan_I.shape[0] * scan_J.shape[0], 2), dtype=int)
    index = 0
    for i in scan_I:
        for j in scan_J:
            i_start = i
            i_end = i + window_size
            j_start = j
            j_end = j + window_size

            window_image = raw_img[i_start:i_end, j_start:j_end]
            window_image = cv2.resize(window_image, (window_size, window_size))
            raw_img_subdivided[index, :, :] = window_image

            positions[index, 0] = i_start
            positions[index, 1] = j_start

            index += 1
    return raw_img_subdivided, positions
