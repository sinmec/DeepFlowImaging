import numpy as np
import cv2
from numba import njit


@njit
def parametrize_anchor_box_properties(anchors, anchor_argmax_ious, labels, ious, bbox_dataset, img):

    # Parametrizing the anchor box properties, see lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)
    # The procedure is more or less described in ../OpenPIV_LOV/src/return_bubble_locations.py

    debug = False

    if debug:
        image = img.copy()
        image *= 255.0
        image_rgb = image[:,:,0]
        image_rgb = image_rgb.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

    # Initialzing the array
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

        h_b   = bbox_dataset[index_gt, 4]
        w_b   = bbox_dataset[index_gt, 3]
        c_x_b = bbox_dataset[index_gt, 0]
        c_y_b = bbox_dataset[index_gt, 1]

        angle_b = bbox_dataset[index_gt, 5]
        d_1_b   = bbox_dataset[index_gt, 6]
        d_2_b   = bbox_dataset[index_gt, 7]

        # if not labels[i] == -1:

        # Only if valid
        if labels[i] == 1:
            d_x = (c_x_b - c_x_a) / w_a
            d_y = (c_y_b - c_y_a) / h_a
            d_w = np.log(w_b / w_a)
            d_h = np.log(h_b / h_a)

            # d_1   = np.arctanh(0.0 + (d_1_b / w_a))
            # d_2   = np.arctanh(0.0 + (d_2_b / h_a))
            # angle = np.log(angle_b / 360.0)

            # print(d_1_b, d_2_b)
            # d_1   = np.log((w_a / d_1_b))
            # d_2   = np.log((h_a / d_2_b))
            # angle = np.log(angle_b / 360.0)

            d_1   = d_1_b / w_a
            d_2   = d_2_b / h_a
            # angle = np.log(angle_b / 360.0)
            angle = angle_b / 360.0

            # print(d_1_b, d_2_b, angle_b)


            # d_w = w_b / w_a
            # d_h = h_b / h_a
            # obj_id = anchor_argmax_ious[i]
            # print(ious[obj_id,i])
            # labels[i] = ious[obj_id,i]


        # if those values are greater than 1000.0 (see losses.py), they do not enter in
        # the gradient descent calculation
        else:
            d_x   = 10000.0
            d_y   = 10000.0
            d_h   = 10000.0
            d_w   = 10000.0
            d_1   = 10000.0
            d_2   = 10000.0
            angle = 10000.0

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





