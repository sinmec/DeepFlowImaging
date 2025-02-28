import os
import sys
from pathlib import Path
import cv2
import h5py as h5
import numpy as np
from tensorflow import keras

sys.path.append("../training/simple")

from divide_image import divide_image
from create_anchors import create_anchors
from FRCNN.training.simple.return_bbox_from_model import return_bbox_from_model

RPN_TOP_SAMPLES = 100000
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.1


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

out_folder = Path(out_folder_name)
out_folder.mkdir(exist_ok=True)

anchors, index_anchors_valid = create_anchors(
    IMG_SIZE, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

raw_images_dir = Path(INPUT_FOLDER, "Validation/images_full")

if MODE == "mask":
    mask_images_dir = Path(INPUT_FOLDER, "Validation/masks_full")
elif MODE == "raw":
    mask_images_dir = raw_images_dir


test_raw_files = os.listdir(raw_images_dir)
test_masks_files = os.listdir(mask_images_dir)

test_raw_files.sort()
test_masks_files.sort()

test_masks_imgs = []
test_raw_imgs = []
for img in test_raw_files:
    _img = cv2.imread(os.path.join(raw_images_dir, img), 0)
    test_raw_imgs.append(_img)
for img in test_masks_files:
    _img = cv2.imread(os.path.join(mask_images_dir, img), 0)
    test_masks_imgs.append(_img)

N_imgs = len(test_raw_imgs)

anchors, index_anchors_valid = create_anchors(
    IMG_SIZE, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

for img_file in test_raw_files:
    img_raw = cv2.imread(os.path.join(raw_images_dir, img_file), 0)
    img_mask = cv2.imread(os.path.join(mask_images_dir, img_file), 0)

    cropped_raw_imgs = []
    cropped_mask_imgs = []

    if MODE == "mask":

        subdivided_imgs, positions = divide_image(
            img_mask, IMG_SIZE[0], stride_division=3
        )
    elif MODE == "raw":
        subdivided_imgs, positions = divide_image(
            img_raw, IMG_SIZE[0], stride_division=3
        )

    bboxes = []
    scores = []
    N_imgs = subdivided_imgs.shape[0]
    imgs = np.zeros((N_imgs, IMG_SIZE[0], IMG_SIZE[0], 1), dtype=np.float64)
    imgs[:, :, :, 0] = subdivided_imgs.astype(np.float64) / 255.0

    inferences = model.predict(imgs)

    for k, sub_img in enumerate(subdivided_imgs):

        img = np.zeros((1, IMG_SIZE[0], IMG_SIZE[0], 1), dtype=np.float64)
        img[0, :, :, 0] = sub_img.astype(np.float64) / 255.0

        _start_i = positions[k, 0]
        _end_i = _start_i + IMG_SIZE[0]
        _start_j = positions[k, 1]
        _end_j = _start_j + IMG_SIZE[1]

        bbox_pred = inferences[0][k]
        labels_pred = inferences[1][k]

        labels_pred_rav = np.ravel(labels_pred)
        bbox_pred_rav = np.ravel(bbox_pred)

        labels_pred_rav_argsort = np.argsort(labels_pred_rav)
        labels_pred_rav_argsort = labels_pred_rav_argsort[-RPN_TOP_SAMPLES:]
        labels_top = labels_pred_rav[labels_pred_rav_argsort]

        for m in range(len(labels_top)):
            k_A = labels_pred_rav_argsort[m]
            label_A = labels_pred_rav[k_A]

            BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav)

            if BBOX_A[0] < 2:
                continue
            if BBOX_A[1] < 2:
                continue
            if BBOX_A[0] + BBOX_A[2] > IMG_SIZE[0] - 2:
                continue
            if BBOX_A[1] + BBOX_A[3] > IMG_SIZE[0] - 2:
                continue

            BBOX_A[0] += _start_j
            BBOX_A[1] += _start_i
            bboxes.append(BBOX_A)
            scores.append(float(label_A))

    nms_indexes = cv2.dnn.NMSBoxes(bboxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD)

    img_raw_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    full_image_size = (img_mask_rgb.shape[0], img_mask_rgb.shape[1])
    for nms in nms_indexes:
        _bbox = bboxes[nms]

        x_1 = _bbox[0]
        y_1 = _bbox[1]
        x_2 = _bbox[0] + _bbox[2]
        y_2 = _bbox[1] + _bbox[3]

        x_1 = max(x_1, 0)
        y_1 = max(y_1, 0)
        x_2 = min(x_2, full_image_size[1])
        y_2 = min(y_2, full_image_size[0])

        cv2.rectangle(img_raw_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)
        cv2.rectangle(img_mask_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)

    cv2.imwrite(
        os.path.join(out_folder, f"{Path(img_file).stem}.jpg"),
        np.hstack((img_raw_rgb, img_mask_rgb)),
    )
