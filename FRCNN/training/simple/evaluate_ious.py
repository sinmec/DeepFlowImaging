import numpy as np
from numba import jit


@jit(nopython=True)
def evaluate_ious(
        anchors,
        index_anchors_valid,
        ious,
        bbox_dataset,
        image,
        POS_IOU_THRESHOLD,
        NEG_IOU_THRESHOLD,
        debug=False,
):
    num_anchors = len(anchors)
    labels = np.full(num_anchors, -1, dtype=np.int32)

    mask = np.zeros(num_anchors, dtype=np.int32)
    mask[index_anchors_valid] = 1

    gt_argmax_ious = np.empty(ious.shape[0], dtype=np.int32)
    gt_max_ious = np.empty(ious.shape[0], dtype=np.float32)

    for i in range(ious.shape[0]):
        gt_argmax_ious[i] = np.argmax(ious[i])
        gt_max_ious[i] = ious[i, gt_argmax_ious[i]]

    anchor_argmax_ious = np.empty(ious.shape[1], dtype=np.int32)
    anchor_max_ious = np.empty(ious.shape[1], dtype=np.float32)

    for j in range(ious.shape[1]):
        anchor_argmax_ious[j] = np.argmax(ious[:, j])
        anchor_max_ious[j] = ious[anchor_argmax_ious[j], j]

    for i in range(num_anchors):
        if anchor_max_ious[i] < NEG_IOU_THRESHOLD:
            labels[i] = 0
        elif anchor_max_ious[i] >= POS_IOU_THRESHOLD:
            labels[i] = 1

    return labels, anchor_argmax_ious

# OLD IMPLEMENTATION --- Useful for debugging

# def evaluate_ious(
#     anchors,
#     index_anchors_valid,
#     ious,
#     bbox_dataset,
#     image,
#     POS_IOU_THRESHOLD,
#     NEG_IOU_THRESHOLD,
#     debug=False,
# ):
#
#     labels = np.zeros(len(anchors), dtype=np.int32)
#     labels[...] = -1
#
#     mask = np.zeros_like(labels)
#     mask[index_anchors_valid] = 1
#
#     if debug:
#         cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#         image = image.copy() * 255.0
#         image_rgb = image.astype(np.uint8)
#         image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
#
#     gt_argmax_ious = ious.argmax(axis=1)
#     gt_max_ious = ious[np.arange(ious.shape[0]), gt_argmax_ious]
#
#     anchor_argmax_ious = ious.argmax(axis=0)
#     anchor_max_ious = ious[anchor_argmax_ious, np.arange(ious.shape[1])]
#
#     labels[anchor_max_ious < NEG_IOU_THRESHOLD] = 0
#     labels[anchor_max_ious >= POS_IOU_THRESHOLD] = 1
#     # labels[gt_argmax_ious] = 1
#
#     if debug:
#         print("max_IOU", np.max(ious))
#         print("ious_shape", ious.shape)
#         # for k in range(len(labels)):
#         # anchor = anchors[k]
#         # obj_id = anchor_argmax_ious[k]
#         # print(ious[obj_id, k], obj_id)
#         # x_a_1 =  int(anchor[0])
#         # y_a_1 =  int(anchor[1])
#         # x_a_2 =  int(anchor[2])
#         # y_a_2 =  int(anchor[3])
#         # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#         # p_1 = (x_a_1, y_a_1)
#         # p_2 = (x_a_2, y_a_2)
#         # print(p_1, p_2)
#         # cv2.rectangle(image_rgb, p_1, p_2, (0,0,255), 1)
#         # # print(gt_max_ious[k])
#         # cv2.imshow("test", image_rgb)
#         # cv2.waitKey(0)
#
#         # debug
#         for k, bbox in enumerate(bbox_dataset):
#             x_b_1 = int(bbox[0] - (bbox[2] / 2))
#             y_b_1 = int(bbox[1] - (bbox[3] / 2))
#             x_b_2 = int(bbox[0] + (bbox[2] / 2))
#             y_b_2 = int(bbox[1] + (bbox[3] / 2))
#             c_x = (x_b_1 + x_b_2) // 2
#             c_y = (y_b_1 + y_b_2) // 2
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 255)
#             thickness = 1
#             # cv2.putText(image_rgb, "%d"%k, (c_x,c_y), font,
#             # fontScale, color, thickness, cv2.LINE_AA)
#
#             p_1 = (x_b_1, y_b_1)
#             p_2 = (x_b_2, y_b_2)
#             cv2.rectangle(image_rgb, p_1, p_2, (0, 255, 255), 4)
#             img_rgb_annotate = image_rgb.copy()
#
#         # cv2.imshow("test", img_rgb_annotate)
#         # cv2.waitKey(0)
#
#         # anchor = anchors[k]
#         # obj_id = anchor_argmax_ious[k]
#
#         for k in range(len(labels)):
#             image_rgb = img_rgb_annotate
#             if labels[k] == 1:
#                 anchor = anchors[k]
#                 obj_id = anchor_argmax_ious[k]
#                 if ious[obj_id, k] > POS_IOU_THRESHOLD:
#                     x_a_1 = int(anchor[0])
#                     y_a_1 = int(anchor[1])
#                     x_a_2 = int(anchor[2])
#                     y_a_2 = int(anchor[3])
#                     p_1 = (x_a_1, y_a_1)
#                     p_2 = (x_a_2, y_a_2)
#                     cv2.rectangle(image_rgb, p_1, p_2, (0, 0, 255), 2)
#
#         cv2.imshow("test", image_rgb)
#         cv2.waitKey(0)
#
#     return labels, anchor_argmax_ious
