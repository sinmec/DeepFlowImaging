import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
import h5py as h5

# Essa eh uma gambi horrorosa!!!! Evite ao maximo fazer algo do tipo!!!
import sys

sys.path.append("../training/simple")

from read_dataset import read_dataset
from create_anchors import create_anchors
from return_bbox_from_model import return_bbox_from_model

model_file = Path("/home/rafaelfc/Data/DeepFlowImaging/FRCNN/training/simple/best_fRCNN_mask_16.keras")
model_config_file = Path("/home/rafaelfc/Data/DeepFlowImaging/FRCNN/training/simple/best_fRCNN_mask_16_CONFIG.h5")

model = keras.models.load_model(model_file, compile=False)

h5_model = h5.File(model_config_file, 'r')

N_SUB         = h5_model.attrs["N_SUB"]
ANCHOR_SIZES      = h5_model.attrs["ANCHOR_SIZES"]
ANCHOR_RATIOS     = h5_model.attrs["ANCHOR_RATIOS"]
IMG_SIZE          = h5_model.attrs["IMG_SIZE"]
MODE              = h5_model.attrs["MODE"]
POS_IOU_THRESHOLD = h5_model.attrs['POS_IOU_THRESHOLD']
NEG_IOU_THRESHOLD = h5_model.attrs['NEG_IOU_THRESHOLD']
h5_model.close()

out_folder = Path('examples_TEST')
out_folder.mkdir(exist_ok=True)

img_size = IMG_SIZE

RPN_top_samples = 100000

dataset_folder = Path("/home/rafaelfc/Data/DeepFlowImaging/FRCNN/examples/example_dataset_FRCNN_PIV_subimage/Output/")
imgs, bbox_datasets, img_raws = read_dataset(img_size, dataset_folder, subset="Verification")

N_imgs = imgs.shape[0]

# Creating the anchors
anchors, index_anchors_valid = create_anchors(img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES)

for k in range(N_imgs):

    img_out = imgs[k] * 255.0
    img_out = img_out.astype(np.uint8)
    img_mask_rgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
    img_raw_rgb = img_raws[k] * 255.0
    img_raw_rgb = img_raw_rgb.astype(np.uint8)
    img_raw_rgb = cv2.cvtColor(img_raw_rgb, cv2.COLOR_GRAY2BGR)

    bbox_dataset = bbox_datasets[k]
    # Drawing bbox dataset
    for _bbox in enumerate(bbox_dataset):
        bbox = _bbox[1]
        x_b_1 = int(bbox[0] - (bbox[2] / 2))
        y_b_1 = int(bbox[1] - (bbox[3] / 2))
        x_b_2 = int(bbox[0] + (bbox[2] / 2))
        y_b_2 = int(bbox[1] + (bbox[3] / 2))
        c_x = (x_b_1 + x_b_2) // 2
        c_y = (y_b_1 + y_b_2) // 2
        p_1 = (x_b_1, y_b_1)
        p_2 = (x_b_2, y_b_2)
        cv2.rectangle(img_mask_rgb, p_1, p_2, (0, 255, 255), 2)
        cv2.rectangle(img_raw_rgb, p_1, p_2, (0, 255, 255), 2)

    # Image in TF/Keras format
    img = np.zeros((1, img_size[0], img_size[1], 1), dtype=np.float64)
    img[0, :, :, 0] = imgs[k]

    # Infering the labels(Acc.) and bbox positions from the RPN/f-RCNN model
    inference = model.predict(img)
    bbox_pred = inference[0]  # bbox leftmost and right most points
    labels_pred = inference[1]  # labels - probability of being foreground (Acc.)

    # Raveling/Flattening the two returned values
    # TODO: I think that I ravel the output because it returns a multi-dimensional list.
    labels_pred_rav = np.ravel(labels_pred)
    bbox_pred_rav = np.ravel(bbox_pred)

    # Retrieving only the N top highest acc. boxes
    labels_pred_rav_argsort = np.argsort(labels_pred_rav)
    labels_pred_rav_argsort = labels_pred_rav_argsort[-RPN_top_samples:]
    labels_top = labels_pred_rav[labels_pred_rav_argsort]

    # The model does not retun the bboxes ax (x1,x2,x3 and x4)
    # Since there is a additional regression in the f-RCNN/RPN model,
    # It actually return the bbox in a scale-invariant fashion
    # (lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)
    # So, I need to transform the outputs to real dimensional pixel values
    bboxes = []
    scores = []
    for m in range(len(labels_top)):
        # k_A is the anchor index
        k_A = labels_pred_rav_argsort[m]

        # label_A its score(acc.)
        label_A = labels_pred_rav[k_A]

        # Returning the bounding boxes points in pixel dimensions
        BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav, labels_pred_rav)

        # Updating the two lists to be used in the IOU OpenCV built-in implementation
        bboxes.append(BBOX_A)
        scores.append(float(label_A))

    print(np.nanmin(scores), np.nanmax(scores))

    # After inserting the bbox and scores to separate lists,
    # I send this data to the this builtin OpenCV Function.
    # In short, the bboxes are all overlaped. This function removes
    # redundancies, preferring bboxes with higer acc.(score) values
    # NMS means Non-Maximum supression.
    # The first value is the score threshold and the second is the NMS value
    # Those values work fine, but you may tune for your application
    nms_indexes = cv2.dnn.NMSBoxes(bboxes, scores, 0.9, 0.1)

    # Now I loop around the nms_indexes to get the bounding boxes
    bboxes = []
    for index in nms_indexes:
        # Exaclty as in the previous loop...
        k_A = labels_pred_rav_argsort[index]
        BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav, labels_pred_rav)

        # Now I am modifying the bounding boxes, for the NMSBoxes functions, the bboxes
        # are not defined through their points..
        x_1 = BBOX_A[0]
        y_1 = BBOX_A[1]
        x_2 = BBOX_A[0] + BBOX_A[2]
        y_2 = BBOX_A[1] + BBOX_A[3]

        # 'Rescaling' the bounding boxes to comply with the original image size
        x_r = 1.0;
        y_r = 1.0
        x_1 = int(x_1 / x_r)
        y_1 = int(y_1 / y_r)
        x_2 = int(x_2 / x_r)
        y_2 = int(y_2 / y_r)

        # Limiting the values to avoid further problems
        x_1 = max(x_1, 0)
        y_1 = max(y_1, 0)
        x_2 = min(x_2, img_size[1])
        y_2 = min(y_2, img_size[0])

        cv2.rectangle(img_mask_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)
        cv2.rectangle(img_raw_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)

        # Appeding to the bboxes list, rectangles containing the bouding boxes
        bboxes.append((x_1, y_1, x_2, y_2))

    # cv2.imshow("debug", img_mask_rgb)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(out_folder, "out_%06d.jpg" % k), np.hstack((img_raw_rgb, img_mask_rgb)))
