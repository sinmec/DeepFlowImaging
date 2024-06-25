import numpy as np
import cv2

def create_samples_for_training(anchors, index_anchors_valid, anchor_argmax_ious, labels, ious, bbox_dataset, image,
                                POS_RATIO, N_SAMPLES, SHOW_N_POS, debug=False):

    if debug:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        image = image.copy() * 255.0
        image_rgb = image.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

    # When labelling the images, most of pixel valus correspond to background.
    # That is also valid for the anchor/samples. In order to improve training,
    # we limit ourselves to a fixed number of "positive" samples (background).
    # Therefore, this function limits the number of "positive" samples, by ran
    # dom selecting samples and selecting a similar number of "negative samples

    # Number of positive samples
    n_pos = int(POS_RATIO * N_SAMPLES)

    # Positive samples are labelled as 1
    pos_index = np.where(labels == 1)[0]

    # If too much positive samples, disable them by setting th flag as -1
    # labels defined as -1 are not used during tranining
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        labels[disable_index] = -1
        # labels[disable_index] = 0

    # Number of negative samples
    n_neg = N_SAMPLES - np.sum(labels == 1)

    if SHOW_N_POS:
        print('n_posi = %d / %d(set)' % (np.sum(labels == 1), POS_RATIO * N_SAMPLES))

    # Disabling part of the negative samples to have a "good" neg/pos sample ratio
    # It is always good to print this value to check for inconsistencies
    neg_index = np.where(labels == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        labels[disable_index] = -1
        # labels[disable_index] = 0

    if debug:
        for k, bbox in enumerate(bbox_dataset):
            x_b_1 = int(bbox[0] - (bbox[3] / 2))
            y_b_1 = int(bbox[1] - (bbox[4] / 2))
            x_b_2 = int(bbox[0] + (bbox[3] / 2))
            y_b_2 = int(bbox[1] + (bbox[4] / 2))
            c_x = (x_b_1 + x_b_2) // 2
            c_y = (y_b_1 + y_b_2) // 2
            p_1 = (x_b_1, y_b_1)
            p_2 = (x_b_2, y_b_2)
            cv2.rectangle(image_rgb, p_1, p_2, (0,255,255), 4)

        for k in range(len(labels)):
            if labels[k] == 1:
                anchor = anchors[k]
                x_a_1 =  int(anchor[0])
                y_a_1 =  int(anchor[1])
                x_a_2 =  int(anchor[2])
                y_a_2 =  int(anchor[3])
                p_1 = (x_a_1, y_a_1)
                p_2 = (x_a_2, y_a_2)
                cv2.rectangle(image_rgb, p_1, p_2, (255,255,0), 4)
            elif labels[k] == 0:
                anchor = anchors[k]
                x_a_1 =  int(anchor[0])
                y_a_1 =  int(anchor[1])
                x_a_2 =  int(anchor[2])
                y_a_2 =  int(anchor[3])
                p_1 = (x_a_1, y_a_1)
                p_2 = (x_a_2, y_a_2)
                cv2.rectangle(image_rgb, p_1, p_2, (0,0,255), 4)

        print('debug')
        cv2.imshow("test", image_rgb)
        cv2.waitKey(1)

    return labels
