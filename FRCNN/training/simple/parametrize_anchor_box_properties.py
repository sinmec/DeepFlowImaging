import numpy as np
import cv2
from numba import njit


@njit
def parametrize_anchor_box_properties(anchors, anchor_argmax_ious, labels, ious, bbox_dataset, img):

    # Parametrizing the anchor box properties,
    # see lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)

    debug = False

    if debug:
        image = img.copy()
        image *= 255.0
        image_rgb = image[:,:,0]
        image_rgb = image_rgb.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

    # Initializing the array
    anchor_locations = np.zeros((len(anchors) * 4), dtype=np.float64)

    # Parametrizing the values
    index = 0
    for i in range(len(labels)):
        x_a_1 = anchors[i,0]
        y_a_1 = anchors[i,1]
        x_a_2 = anchors[i,2]
        y_a_2 = anchors[i,3]

        w_a = x_a_2 - x_a_1
        h_a = y_a_2 - y_a_1
        c_x_a = x_a_1 + 0.5 * w_a
        c_y_a = y_a_1 + 0.5 * h_a

        index_gt = anchor_argmax_ious[i]


        c_x_b = bbox_dataset[index_gt, 0]
        c_y_b = bbox_dataset[index_gt, 1]
        h_b = bbox_dataset[index_gt, 2]
        w_b = bbox_dataset[index_gt, 3]

        if labels[i] == 1:
            d_x = (c_x_b - c_x_a) / w_a
            d_y = (c_y_b - c_y_a) / h_a
            d_w = np.log(w_b / w_a)
            d_h = np.log(h_b / h_a)

        else:
            d_x   = 10000.0
            d_y   = 10000.0
            d_h   = 10000.0
            d_w   = 10000.0

        anchor_locations[index]     = d_x
        anchor_locations[index + 1] = d_y
        anchor_locations[index + 2] = d_h
        anchor_locations[index + 3] = d_w
        #anchor_locations[index + 4]     = angle
        #anchor_locations[index + 5] = d_1
        #anchor_locations[index + 6] = d_2


        index += 4
        # It jumps 4 because it is a single array


    return anchor_locations





