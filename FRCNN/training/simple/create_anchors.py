import numpy as np


def create_anchors(img_size, subsampling_ratio, anchor_ratios, anchor_sizes):
    # The number of anchors points is defined by the subsampling ratio
    N_x = img_size[0] // subsampling_ratio
    N_y = img_size[1] // subsampling_ratio

    # Creating an array to store all possible anchors
    anchors = np.zeros((N_x * N_y * len(anchor_sizes) * len(anchor_ratios), 4))

    # Defining the anchor points position
    c_x_s = np.arange(subsampling_ratio, (N_y + 1) * (subsampling_ratio), subsampling_ratio)
    c_y_s = np.arange(subsampling_ratio, (N_x + 1) * (subsampling_ratio), subsampling_ratio)

    # Defining the anchor points leftmost and rightmost points
    # X----------
    # |         |
    # |         |
    # |         |
    # ----------X
    index = 0
    for c_y in c_y_s:
        for c_x in c_x_s:
            for i in range(len(anchor_sizes)):
                for j in range(len(anchor_ratios)):
                    h = anchor_sizes[i] * np.sqrt(anchor_ratios[j]) * subsampling_ratio
                    w = anchor_sizes[i] * np.sqrt(1.0 / anchor_ratios[j]) * subsampling_ratio
                    h = int(h)
                    w = int(w)

                    x_1 = c_x - (w / 2)
                    y_1 = c_y - (h / 2)
                    x_2 = c_x + (w / 2)
                    y_2 = c_y + (h / 2)

                    anchors[index, 0] = x_1
                    anchors[index, 1] = y_1
                    anchors[index, 2] = x_2
                    anchors[index, 3] = y_2

                    index += 1

    # Removing anchors that are outside the image
    index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= img_size[1]) &
        (anchors[:, 3] <= img_size[0])
    )[0]

    return anchors, index_inside
