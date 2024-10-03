import numpy as np
from numba import njit


@njit
def calculate_bbox_intesect_over_union(
    anchors, index_anchors_valid, bbox_dataset, image
):
    ious = np.zeros((len(bbox_dataset), len(anchors)), dtype=np.float32)

    valid_anchors = anchors[index_anchors_valid]

    for k in range(len(bbox_dataset)):

        bbox = bbox_dataset[k]
        x_b_1 = int(bbox[0] - (bbox[2] / 2))
        y_b_1 = int(bbox[1] - (bbox[3] / 2))
        x_b_2 = int(bbox[0] + (bbox[2] / 2))
        y_b_2 = int(bbox[1] + (bbox[3] / 2))

        bbox_area = (x_b_2 - x_b_1) * (y_b_2 - y_b_1)
        for i, valid_anchor in enumerate(valid_anchors):
            x_a_1 = int(valid_anchor[0])
            y_a_1 = int(valid_anchor[1])
            x_a_2 = int(valid_anchor[2])
            y_a_2 = int(valid_anchor[3])

            anchor_area = (x_a_2 - x_a_1) * (y_a_2 - y_a_1)

            xA = max(x_a_1, x_b_1)
            yA = max(y_a_1, y_b_1)
            xB = min(x_a_2, x_b_2)
            yB = min(y_a_2, y_b_2)

            inter_area = max(0, xB - xA) * max(0, yB - yA)
            iou = float(inter_area) / float(bbox_area + anchor_area - inter_area)
            index = index_anchors_valid[i]
            ious[k, index] = iou

    return ious
