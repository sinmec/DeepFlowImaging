import os
import sys
from pathlib import Path
import cv2
import h5py as h5
import numpy as np
from tensorflow import keras

sys.path.append("../training/simple")

from divide_image import divide_image
from recreate_image import recreate_image
from create_anchors import create_anchors
from FRCNN.training.simple.return_bbox_from_model import return_bbox_from_model

model_file = Path(
    "../training/simple/best_fRCNN_mask_16.keras"
)

model_config_file = Path(
    "../training/simple/best_fRCNN_mask_16_CONFIG.h5"
)

model = keras.models.load_model(model_file, compile=False)

with h5.File(model_config_file, "r") as h5_model:
    N_SUB = h5_model.attrs["N_SUB"]
    ANCHOR_SIZES = h5_model.attrs["ANCHOR_SIZES"]
    ANCHOR_RATIOS = h5_model.attrs["ANCHOR_RATIOS"]
    IMG_SIZE = h5_model.attrs["IMG_SIZE"]
    POS_IOU_THRESHOLD = h5_model.attrs["POS_IOU_THRESHOLD"]
    NEG_IOU_THRESHOLD = h5_model.attrs["NEG_IOU_THRESHOLD"]

out_folder = Path("examples_TEST_full_image")
out_folder.mkdir(exist_ok=True)

img_size = IMG_SIZE

RPN_top_samples = 100000

anchors, index_anchors_valid = create_anchors(
    img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

# Folder where the test images are stored
raw_images_dir = Path(
    "/home/higorem/DeepFlowImaging/FRCNN/examples/example_dataset_FRCNN_PIV_subimage_TEST/Output/Validation/images_full")

mask_images_dir = Path(
    "/home/higorem/DeepFlowImaging/FRCNN/examples/example_dataset_FRCNN_PIV_subimage_TEST/Output/Validation/masks_full")

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
    img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

for img_file in test_raw_files:
    img_raw_file = cv2.imread(os.path.join(raw_images_dir, img_file), 0)
    img_mask_file = cv2.imread(os.path.join(mask_images_dir, f"UNET_{img_file}"), 0)

    cropped_raw_imgs = []
    cropped_mask_imgs = []

    subdivide_images_mask = divide_image(img_mask_file, 512)
    subdivide_images_raw = divide_image(img_raw_file, 512)

    pos_sub_imgs_mask = subdivide_images_mask[1]
    sub_imgs_mask = subdivide_images_mask[0]
    pos_sub_imgs_raw = subdivide_images_raw[1]
    sub_imgs_raw = subdivide_images_raw[0]

    for k, subimg in enumerate(sub_imgs_mask):
        img_out_raw = sub_imgs_raw[k] * 255.0
        img_out_raw = img_out_raw.astype(np.uint8)
        img_raw_rgb = cv2.cvtColor(img_out_raw, cv2.COLOR_GRAY2BGR)

        img_out_mask = sub_imgs_mask[k] * 255.0
        img_out_mask = img_out_mask.astype(np.uint8)
        img_mask_rgb = cv2.cvtColor(img_out_mask, cv2.COLOR_GRAY2BGR)

        img = np.zeros((1, 512, 512, 1), dtype=np.float64)
        img[0, :, :, 0] = subimg

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
        for nms in nms_indexes:
            k_A = labels_pred_rav_argsort[nms]
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

            cv2.rectangle(img_raw_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)
            cv2.rectangle(img_mask_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)
            bboxes.append((x_1, y_1, x_2, y_2))

        cropped_raw_imgs.append(img_raw_rgb)
        cropped_mask_imgs.append(img_mask_rgb)

    raw_img = recreate_image(cropped_raw_imgs, test_raw_imgs[0].shape, pos_sub_imgs_raw)
    mask_img = recreate_image(cropped_mask_imgs, test_raw_imgs[0].shape, pos_sub_imgs_raw)

    img_out = np.hstack((raw_img, mask_img))

    cv2.imwrite(os.path.join(out_folder, f"FULL_{Path(img_file).stem}.jpg"), img_out)
