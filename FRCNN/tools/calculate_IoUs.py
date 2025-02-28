import numpy as np
from return_bounding_box_points import return_bounding_box_points


# @njit
def calculate_IoUs(bboxes_dataset, bboxes_predicted):

    # Initializing IoUs array
    ious = np.zeros((len(bboxes_predicted), len(bboxes_dataset)), dtype=np.float32)

    # Looping over the dataset
    # and calculating IoU of each predicted bbox against all the bbox_datasets
    for i in range(len(bboxes_predicted)):
        bbox = bboxes_predicted[i]
        x_p_1 = bbox[0]
        y_p_1 = bbox[1]
        x_p_2 = bbox[2]
        y_p_2 = bbox[3]
        bbox_pred_area = (x_p_2 - x_p_1) * (y_p_2 - y_p_1)

        # Looping over the bboxes from the dataset
        for j in range(len(bboxes_dataset)):
            bbox = bboxes_dataset[j]

            p_1, p_2 = return_bounding_box_points(bbox)
            # TODO: FIXIT!
            x_d_1 = p_1[0]
            y_d_1 = p_1[1]
            x_d_2 = p_2[0]
            y_d_2 = p_2[1]
            bbox_dset_area = (x_d_2 - x_d_1) * (y_d_2 - y_d_1)

            # Finding intersection points
            xA = max(x_p_1, x_d_1)
            yA = max(y_p_1, y_d_1)
            xB = min(x_p_2, x_d_2)
            yB = min(y_p_2, y_d_2)

            # Computing intersection area
            inter_area = max(0, xB - xA) * max(0, yB - yA)

            # Calculating IoU for given dataset/pred pair
            iou = float(inter_area) / float(
                bbox_pred_area + bbox_dset_area - inter_area
            )

            # Updating IoU array
            ious[i, j] = iou

    return ious
