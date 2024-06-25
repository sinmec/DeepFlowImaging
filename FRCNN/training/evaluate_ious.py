import numpy as np
import cv2


def evaluate_ious(anchors, index_anchors_valid, ious, bbox_dataset, image,
                 POS_IOU_THRESHOLD,  NEG_IOU_THRESHOLD, debug=False):


    # Initializing labels with a -1 value
    labels = np.zeros(len(anchors), dtype=np.int32)
    labels[...] = -1

    # Mask is the anchors which are valid
    # It seems that I am not doing any distinction
    # betweetn valid and invalid anchors
    mask = np.zeros_like(labels)
    mask[index_anchors_valid] = 1

    if debug:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        image = image.copy() * 255.0
        image_rgb = image.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)


    # index of the max IOUs from a given ground-truth object
    gt_argmax_ious = ious.argmax(axis=1)
    gt_max_ious = ious[np.arange(ious.shape[0]), gt_argmax_ious]

    # index of the max anchor box IOU for a given ground-truth object
    anchor_argmax_ious = ious.argmax(axis=0)
    anchor_max_ious = ious[anchor_argmax_ious, np.arange(ious.shape[1])]

    # If the label has a low a IoU value, it is labelled as background
    # On the contrary, it is labelled as foreground
    labels[anchor_max_ious < NEG_IOU_THRESHOLD] = 0
    labels[anchor_max_ious >= POS_IOU_THRESHOLD] = 1
    # labels[gt_argmax_ious] = 1

    #TODO: Check values (0, 1, -1)

    # Here I was applying the mask
    # index_outside = mask[mask == 0]
    # labels[index_outside] = -1


    # From down here it is only debugging functions..
    if debug:
        print("max_IOU", np.max(ious))
        print("ious_shape", ious.shape)
        # for k in range(len(labels)):
           # anchor = anchors[k]
           # obj_id = anchor_argmax_ious[k]
           # print(ious[obj_id, k], obj_id)
           # x_a_1 =  int(anchor[0])
           # y_a_1 =  int(anchor[1])
           # x_a_2 =  int(anchor[2])
           # y_a_2 =  int(anchor[3])
           # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
           # p_1 = (x_a_1, y_a_1)
           # p_2 = (x_a_2, y_a_2)
           # print(p_1, p_2)
           # cv2.rectangle(image_rgb, p_1, p_2, (0,0,255), 1)
           # # print(gt_max_ious[k])
           # cv2.imshow("test", image_rgb)
           # cv2.waitKey(0)


        # debug
        for k, bbox in enumerate(bbox_dataset):
            x_b_1 = int(bbox[0] - (bbox[3] / 2))
            y_b_1 = int(bbox[1] - (bbox[4] / 2))
            x_b_2 = int(bbox[0] + (bbox[3] / 2))
            y_b_2 = int(bbox[1] + (bbox[4] / 2))
            c_x = (x_b_1 + x_b_2) // 2
            c_y = (y_b_1 + y_b_2) // 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 255)
            thickness = 1
            # cv2.putText(image_rgb, "%d"%k, (c_x,c_y), font,
                       # fontScale, color, thickness, cv2.LINE_AA)

            p_1 = (x_b_1, y_b_1)
            p_2 = (x_b_2, y_b_2)
            cv2.rectangle(image_rgb, p_1, p_2, (0,255,255), 4)
            img_rgb_annotate = image_rgb.copy()

        # cv2.imshow("test", img_rgb_annotate)
        # cv2.waitKey(0)

           # anchor = anchors[k]
           # obj_id = anchor_argmax_ious[k]

        for k in range(len(labels)):
            image_rgb = img_rgb_annotate
            if labels[k] == 1:
                anchor = anchors[k]
                obj_id = anchor_argmax_ious[k]
                if ious[obj_id, k] > POS_IOU_THRESHOLD:
                    x_a_1 =  int(anchor[0])
                    y_a_1 =  int(anchor[1])
                    x_a_2 =  int(anchor[2])
                    y_a_2 =  int(anchor[3])
                    p_1 = (x_a_1, y_a_1)
                    p_2 = (x_a_2, y_a_2)
                    cv2.rectangle(image_rgb, p_1, p_2, (0,0,255), 2)

        cv2.imshow("test", image_rgb)
        cv2.waitKey(0)

    return labels, anchor_argmax_ious
