import os
import sys
from pathlib import Path

import cv2
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.append("../training/simple")

from return_bounding_box_points import return_bounding_box_points

from read_dataset import read_dataset
from create_anchors import create_anchors
from return_bbox_from_model import return_bbox_from_model
from calculate_IoUs import calculate_IoUs

# from calculate_bbox_intesect_over_union import calculate_bbox_intesect_over_union
# from evaluate_ious import evaluate_ious
# from create_samples_for_training import create_samples_for_training
# from parametrize_anchor_box_properties import parametrize_anchor_box_properties

# Defining h5 model file
# model_file = "../best_fRCNN_mask_16_100.0_04.h5"
# model_file = sys.argv[1]

model_file = Path("../training/simple/best_fRCNN_raw_08.keras")

model_config_file = Path("../training/simple/best_fRCNN_raw_08_CONFIG.h5")

out_folder_name = "examples_TEST_2025_FULL"

model = keras.models.load_model(model_file, compile=False)

with h5.File(model_config_file, "r") as h5_model:
    N_SUB = h5_model.attrs["N_SUB"]
    ANCHOR_SIZES = h5_model.attrs["ANCHOR_SIZES"]
    ANCHOR_RATIOS = h5_model.attrs["ANCHOR_RATIOS"]
    IMG_SIZE = h5_model.attrs["IMG_SIZE"]
    MODE = h5_model.attrs["MODE"]
    POS_IOU_THRESHOLD = h5_model.attrs["POS_IOU_THRESHOLD"]
    NEG_IOU_THRESHOLD = h5_model.attrs["NEG_IOU_THRESHOLD"]
    INPUT_FOLDER = h5_model.attrs["INPUT_FOLDER"]

out_folder = Path(out_folder_name, "metrics")
out_folder.mkdir(exist_ok=True)

RPN_TOP_SAMPLES = 100000
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.1

dataset_folder = Path(INPUT_FOLDER)
imgs_mask, bbox_datasets, imgs_raw = read_dataset(
    IMG_SIZE, dataset_folder, subset="Verification"
)

if MODE == "raw":
    imgs = imgs_raw
elif MODE == "mask":
    imgs = imgs_mask

N_imgs = imgs.shape[0]

anchors, index_anchors_valid = create_anchors(
    IMG_SIZE, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

# Number of Ground-truths in each image
N_GTs = np.zeros(N_imgs, dtype=float)
for i in range(N_imgs):
    N_GTs[i] = len(bbox_datasets[i])

# IoU treshold for mAP
# IoU_treshold = 0.5 + 1.0e-3

IoU_thresh_s = np.arange(0.5, 1.0, 0.05)


# Array which store all the bbox information
# for computing the mean-average-precision(mAP)
# [img_index, score, max_IOU, x_1, y_1, x_2, y_2]
bboxes_ALL = []

mAPs = []
for n in range(len(IoU_thresh_s)):

    # IoU threshold from the array
    IoU_treshold = IoU_thresh_s[n]

    for k in range(N_imgs):

        img_out = imgs[k] * 255.0
        img_out = img_out.astype(np.uint8)
        img_mask_rgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        rec_bboxs_datasets = []
        bbox_dataset = bbox_datasets[k]
        # Drawing bbox dataset
        for _bbox in enumerate(bbox_dataset):
            bbox = _bbox[1]
            # x_b_1 = int(bbox[0] - (bbox[3] / 2))
            # y_b_1 = int(bbox[1] - (bbox[4] / 2))
            # x_b_2 = int(bbox[0] + (bbox[3] / 2))
            # y_b_2 = int(bbox[1] + (bbox[4] / 2))
            p_1, p_2 = return_bounding_box_points(bbox)
            x_b_1 = p_1[0]
            y_b_1 = p_1[1]
            x_b_2 = p_2[0]
            y_b_2 = p_2[1]
            c_x = (x_b_1 + x_b_2) // 2
            c_y = (y_b_1 + y_b_2) // 2
            p_1 = (x_b_1, y_b_1)
            p_2 = (x_b_2, y_b_2)
            rec_bboxs_datasets.append((x_b_1, y_b_1, x_b_2, y_b_2))
            cv2.rectangle(img_mask_rgb, p_1, p_2, (0, 255, 255), 2)

        # Image in TF/Keras format
        img = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float)
        img[0, :, :, 0] = imgs[k]

        # Inferring the labels(Acc.) and bbox positions from the RPN/f-RCNN model
        inference = model.predict(img)
        bbox_pred = (inference[0],)
        labels_pred = inference[1]

        # Raveling/Flattening the two returned values
        labels_pred_rav = np.ravel(labels_pred)
        bbox_pred_rav = np.ravel(bbox_pred)

        # Retrieving only the N top highest acc. boxes
        labels_pred_rav_argsort = np.argsort(labels_pred_rav)
        labels_pred_rav_argsort = labels_pred_rav_argsort[-RPN_TOP_SAMPLES:]
        labels_top = labels_pred_rav[labels_pred_rav_argsort]

        # The model does not retun the bboxes ax (x1,x2,x3 and x4)
        # Since there is a additional regression in the f-RCNN/RPN model,
        # It actually return the bbox in a scale-invariant fashion
        # (lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)
        # So, I need to transform the outputs to real dimensional pixel values
        bboxes = []
        scores = []
        for m in range(len(labels_top)):
            k_A = labels_pred_rav_argsort[m]

            label_A = labels_pred_rav[k_A]

            BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav)

            # Updating the two lists to be used in the IOU OpenCV built-in implementation
            bboxes.append(BBOX_A)
            scores.append(float(label_A))

        # After inserting the bbox and scores to separate lists,
        # I send this data to the this builtin OpenCV Function.
        # In short, the bboxes are all overlaped. This function removes
        # redundancies, preferring bboxes with higer acc.(score) values
        # NMS means Non-Maximum supression.
        # The first value is the score threshold and the second is the NMS value
        # Those values work fine, but you may tune for your application
        nms_indexes = cv2.dnn.NMSBoxes(bboxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD)
        # nms_indexes = cv2.dnn.NMSBoxes(bboxes, scores, IoU_treshold, 0.1)

        # Now I loop around the nms_indexes to get the bounding boxes
        bboxes = []

        # Retrieving the scores from the bboxes that survived the NMS
        scores_ALL = []
        for index in nms_indexes:
            scores_ALL.append(scores[index])

        # print(scores_ALL)
        # exit()

        for index in nms_indexes:
            # Exaclty as in the previous loop...
            k_A = labels_pred_rav_argsort[index]
            BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav)

            # Now I am modifying the bounding boxes, for the NMSBoxes functions, the bboxes
            # are not defined through their points..
            x_1 = BBOX_A[0]
            y_1 = BBOX_A[1]
            x_2 = BBOX_A[0] + BBOX_A[2]
            y_2 = BBOX_A[1] + BBOX_A[3]

            # 'Rescaling' the bounding boxes to comply with the original image size
            x_r = 1.0
            y_r = 1.0
            x_1 = int(x_1 / x_r)
            y_1 = int(y_1 / y_r)
            x_2 = int(x_2 / x_r)
            y_2 = int(y_2 / y_r)

            # Limiting the values to avoid further problems
            x_1 = max(x_1, 0)
            y_1 = max(y_1, 0)
            x_2 = min(x_2, IMG_SIZE[1])
            y_2 = min(y_2, IMG_SIZE[0])

            # Appeding to the bboxes list, rectangles containing the bouding boxes
            bboxes.append((x_1, y_1, x_2, y_2))

        # Calculating IoUs
        IoUs = calculate_IoUs(bbox_dataset, bboxes)

        # Retrieving the maximum IoU associated with each bounding box
        # In addition, we are associante each pred. bbox with
        # its correspoonding GT box. Multiple associations may occur.
        # When that is the case, we choose the one with highest IoU
        IOU_bbox_index = np.zeros(len(bbox_dataset), dtype=int)
        IOU_bbox_index[...] = -1  # Negative, then we can find non-linked GT bboxes
        IOU_bbox_score = np.zeros(len(bbox_dataset), dtype=float)
        IoUs_max = np.zeros(len(bboxes), dtype=float)
        for i in range(len(bboxes)):
            IoU_max = np.nanmax(IoUs[i, :])
            IoUs_max[i] = IoU_max
            idx_bbox_GT = np.nanargmax(IoUs[i, :])
            if IoU_max >= IOU_bbox_score[idx_bbox_GT]:
                IOU_bbox_index[idx_bbox_GT] = i
                IOU_bbox_score[idx_bbox_GT] = IoU_max

        # Negating the IoU from the 'multiplicity' check
        for i in range(len(bboxes)):
            if i not in IOU_bbox_index:
                IoUs_max[i] *= -1.0

        #    #DEBUG 4
        #    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        #    _img_debug = img_mask_rgb.copy()
        #    for i in range(len(IOU_bbox_score)):
        #        if (IOU_bbox_index[i] < 0):
        #            continue
        #
        #        # if IOU_bbox_score[i] > -IoU_treshold:
        #        idx_bbox_pred = IOU_bbox_index[i]
        #        print(idx_bbox_pred)
        #
        #        (x_1, y_1, x_2, y_2) = bboxes[idx_bbox_pred]
        #        cv2.rectangle(_img_debug, (x_1,y_1), (x_2, y_2), (000,255,000), 4)
        #
        #    for i in range(len(bboxes)):
        #        if IoUs_max[i] < 0.0:
        #            (x_1, y_1, x_2, y_2) = bboxes[i]
        #            cv2.rectangle(_img_debug, (x_1,y_1), (x_2, y_2), (000,000,255), 1)
        #
        #    cv2.imshow("debug", _img_debug)
        #    cv2.waitKey(0)

        # Updating the "ALL" array to compute the mAP
        for i in range(len(bboxes)):
            (x_1, y_1, x_2, y_2) = bboxes[i]
            bboxes_ALL.append((k, scores_ALL[i], IoUs_max[i], x_1, y_1, x_2, y_2))

    # Now, with all the informatin from the test data,
    # we can compute the mAP
    # First, we order the predicted bboxes by score
    bboxes_ALL.sort(key=lambda x: x[1], reverse=True)
    # for k in range(len(bboxes_ALL)):
    # print(bboxes_ALL[k][1])

    # Array to store number of True Positives
    TPs = np.zeros(len(bboxes_ALL), dtype=float)

    # Array to store number of False Positives
    FPs = np.zeros(len(bboxes_ALL), dtype=float)

    precision_s = []
    recall_s = []
    # Calculating Precision x Recall curve
    for k in range(len(bboxes_ALL)):
        img_index = bboxes_ALL[k][0]
        score = bboxes_ALL[k][1]
        iou = bboxes_ALL[k][2]
        x_1 = bboxes_ALL[k][3]
        y_1 = bboxes_ALL[k][4]
        x_2 = bboxes_ALL[k][5]
        y_2 = bboxes_ALL[k][6]

        if iou >= IoU_treshold:
            TPs[k] = 1.0
        else:
            FPs[k] = 1.0

        # Calculating precision and recall
        # precision = TP_cumsum[-1] / (TP_cumsum[-1] + FP_cumsum[-1])
        # recall    = TP_cumsum[-1] / np.sum(N_GTs)

        # print(precision, recall)

        # Appending to an array
        # precision_s.append(precision)
        # recall_s.append(recall)

    # Cummulative sum of TPs and FPs
    TP_cumsum = np.cumsum(TPs)
    FP_cumsum = np.cumsum(FPs)
    precision_s = TP_cumsum / (TP_cumsum + FP_cumsum)
    recall_s = TP_cumsum / np.sum(N_GTs)

    print("N_GT", np.sum(N_GTs))
    print("N_TPs", sum(TPs))
    print("N_FPs", sum(FPs))

    # Now we "trapezify" the recall x precision curve
    precision_s_trapz = []
    for j in range(precision_s.shape[0]):
        precision_s_trapz.append(np.nanmax(precision_s[j:]))

    # Now we measure AP-11 (with 11 points)
    recall_s_11 = np.linspace(0.0, 1.0, num=11)
    precision_s_11 = np.zeros(11, dtype=float)
    for i in range(11):
        if recall_s_11[i] <= np.nanmax(recall_s):
            idx_11 = np.argmin(np.abs(recall_s - recall_s_11[i]))
            precision_s_11[i] = precision_s_trapz[idx_11]

    # Now we calculate the Area Under Curve (AUC),
    # and finally retrieve our metric
    AP = np.sum(precision_s_11) / 11.0

    print(f"OUT: {model_file.stem}, {IoU_treshold:.2f}, {AP:.2f}")

    plt.plot(recall_s, precision_s, color="k")
    plt.plot(recall_s, precision_s_trapz, color="r")
    plt.scatter(recall_s_11, precision_s_11, color="blue")
    plt.xlim(0.0, 1.1)
    plt.ylim(0.0, 1.1)
    plt.xlabel("Recall [-]")
    plt.ylabel("Precision [-]")
    plt.title("IoU = %.2f" % IoU_treshold)
    plt.savefig(os.path.join(out_folder, "mAP_IoU_%.2f.png" % IoU_treshold))
    plt.close()

    mAPs.append(AP)


print(f"OUT-final: {model_file.stem} {np.mean(mAPs):.2f}")


# DEBUG 3
# for k in range(len(bboxes_ALL)):
#    img_index = bboxes_ALL[k][0]
#    score = bboxes_ALL[k][1]
#    x_1 = bboxes_ALL[k][3]
#    y_1 = bboxes_ALL[k][4]
#    x_2 = bboxes_ALL[k][5]
#    y_2 = bboxes_ALL[k][6]
#
#   if score > 0.9:
#       img_out = imgs[img_index] * 255.0
#       img_out = img_out.astype(np.uint8)
#       img_mask_rgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
#       cv2.rectangle(img_mask_rgb, (x_1,y_1), (x_2, y_2), (000,255,000), 4)
#       cv2.imshow("debug", img_mask_rgb)
#       cv2.waitKey(0)


#    #DEBUG 2
#    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
#    for i in range(len(bboxes)):
#        _img_debug = img_mask_rgb.copy()
#
#        (x_1, y_1, x_2, y_2) = bboxes[i]
#        cv2.rectangle(_img_debug, (x_1,y_1), (x_2, y_2), (000,255,000), 4)
#
#        j_max = np.nanargmax(IoUs[i,:])
#        __img_debug = _img_debug.copy()
#        (x_1, y_1, x_2, y_2) = rec_bboxs_datasets[j_max]
#        cv2.rectangle(__img_debug, (x_1, y_1), (x_2, y_2), (0,0,255), 3)
#
#        cv2.imshow("debug", __img_debug)
#        print(IoUs[i,j_max])
#        cv2.waitKey(0)

#    #DEBUG 1
#    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
#    for i in range(len(bboxes)):
#        _img_debug = img_mask_rgb.copy()
#
#        (x_1, y_1, x_2, y_2) = bboxes[i]
#        cv2.rectangle(_img_debug, (x_1,y_1), (x_2, y_2), (000,255,000), 4)
#
#        for j in range(len(rec_bboxs_datasets)):
#            __img_debug = _img_debug.copy()
#            (x_1, y_1, x_2, y_2) = rec_bboxs_datasets[j]
#            cv2.rectangle(__img_debug, (x_1, y_1), (x_2, y_2), (0,0,255), 3)
#
#            cv2.imshow("debug", __img_debug)
#            print(IoUs[i,j])
#            cv2.waitKey(0)
