import numpy as np
import cv2


def return_bbox_from_model(k_A, anchors, bbox_pred_rav, labels_rav):

    # RPN model outputs
    d_x_A = bbox_pred_rav[k_A*4 + 0]
    d_y_A = bbox_pred_rav[k_A*4 + 1]
    d_h_A = bbox_pred_rav[k_A*4 + 2]
    d_w_A = bbox_pred_rav[k_A*4 + 3]

    # Real anchor positions
    anchor = anchors[k_A]
    x_a_1 = anchor[0]
    y_a_1 = anchor[1]
    x_a_2 = anchor[2]
    y_a_2 = anchor[3]

    # From scale-invariant parameter to real pixel dimensions
    w_a = x_a_2 - x_a_1
    h_a = y_a_2 - y_a_1
    c_x_a = x_a_1 + 0.5 * w_a
    c_y_a = y_a_1 + 0.5 * h_a

    c_x = (w_a * d_x_A) + c_x_a
    c_y = (h_a * d_y_A) + c_y_a
    h   = np.exp(d_h_A) * h_a
    w   = np.exp(d_w_A) * w_a

    # Converting to integers
    x_1_A = int(c_x - 0.5 * 1.0 * w)
    x_2_A = int(c_x + 0.5 * 1.0 * w)
    y_1_A = int(c_y - 0.5 * 1.0 * h)
    y_2_A = int(c_y + 0.5 * 1.0 * h)


    BBOX_A = [x_1_A, y_1_A, x_2_A - x_1_A, y_2_A - y_1_A]

    return BBOX_A
