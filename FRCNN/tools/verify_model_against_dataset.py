import os

# Essa eh uma gambi horrorosa!!!! Evite ao maximo fazer algo do tipo!!!
import sys
from pathlib import Path

import cv2
import h5py as h5
import numpy as np
from tensorflow import keras

sys.path.append("../training/simple")

from read_dataset import read_dataset
from create_anchors import create_anchors
from return_bbox_from_model import return_bbox_from_model

model_file = Path(
    "../training/simple/best_fRCNN_mask_16.keras"
)
model_config_file = Path(
    "../training/simple/best_fRCNN_mask_16_CONFIG.h5"
)

model = keras.models.load_model(model_file, compile=False)

h5_model = h5.File(model_config_file, "r")

N_SUB = h5_model.attrs["N_SUB"]
ANCHOR_SIZES = h5_model.attrs["ANCHOR_SIZES"]
ANCHOR_RATIOS = h5_model.attrs["ANCHOR_RATIOS"]
IMG_SIZE = h5_model.attrs["IMG_SIZE"]
MODE = h5_model.attrs["MODE"]
POS_IOU_THRESHOLD = h5_model.attrs["POS_IOU_THRESHOLD"]
NEG_IOU_THRESHOLD = h5_model.attrs["NEG_IOU_THRESHOLD"]
h5_model.close()

out_folder = Path("examples_TEST")
out_folder.mkdir(exist_ok=True)

img_size = IMG_SIZE

RPN_top_samples = 100000

dataset_folder = Path(
    "/home/rafaelfc/Data/DATASETS/example_dataset_FRCNN_PIV_subimage/Output"
)
imgs, bbox_datasets, img_raws = read_dataset(
    img_size, dataset_folder, subset="Verification"
)

N_imgs = imgs.shape[0]

anchors, index_anchors_valid = create_anchors(
    img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

for k in range(N_imgs):

    img_out = imgs[k] * 255.0
    img_out = img_out.astype(np.uint8)
    img_mask_rgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
    img_raw_rgb = img_raws[k] * 255.0
    img_raw_rgb = img_raw_rgb.astype(np.uint8)
    img_raw_rgb = cv2.cvtColor(img_raw_rgb, cv2.COLOR_GRAY2BGR)

    bbox_dataset = bbox_datasets[k]
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

    img = np.zeros((1, img_size[0], img_size[1], 1), dtype=np.float64)
    img[0, :, :, 0] = imgs[k]

    inference = model.predict(img)
    bbox_pred = inference[0]
    labels_pred = inference[1]

    labels_pred_rav = np.ravel(labels_pred)
    bbox_pred_rav = np.ravel(bbox_pred)

    labels_pred_rav_argsort = np.argsort(labels_pred_rav)
    labels_pred_rav_argsort = labels_pred_rav_argsort[-RPN_top_samples:]
    labels_top = labels_pred_rav[labels_pred_rav_argsort]

    bboxes = []
    scores = []
    for m in range(len(labels_top)):
        k_A = labels_pred_rav_argsort[m]

        label_A = labels_pred_rav[k_A]

        BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav)

        bboxes.append(BBOX_A)
        scores.append(float(label_A))

    print(np.nanmin(scores), np.nanmax(scores))

    nms_indexes = cv2.dnn.NMSBoxes(bboxes, scores, 0.9, 0.1)

    bboxes = []
    for index in nms_indexes:
        k_A = labels_pred_rav_argsort[index]
        BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav)

        x_1 = BBOX_A[0]
        y_1 = BBOX_A[1]
        x_2 = BBOX_A[0] + BBOX_A[2]
        y_2 = BBOX_A[1] + BBOX_A[3]

        x_r = 1.0
        y_r = 1.0
        x_1 = int(x_1 / x_r)
        y_1 = int(y_1 / y_r)
        x_2 = int(x_2 / x_r)
        y_2 = int(y_2 / y_r)

        x_1 = max(x_1, 0)
        y_1 = max(y_1, 0)
        x_2 = min(x_2, img_size[1])
        y_2 = min(y_2, img_size[0])

        cv2.rectangle(img_mask_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)
        cv2.rectangle(img_raw_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)

        bboxes.append((x_1, y_1, x_2, y_2))

    cv2.imwrite(
        os.path.join(out_folder, "out_%06d.jpg" % k),
        np.hstack((img_raw_rgb, img_mask_rgb)),
    )
