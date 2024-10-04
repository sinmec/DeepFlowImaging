import os

# Essa eh uma gambi horrorosa!!!! Evite ao maximo fazer algo do tipo!!!
import sys
from pathlib import Path

import cv2
import h5py as h5
import numpy as np
from tensorflow import keras

sys.path.append("../training/simple")

from create_anchors import create_anchors
from return_bbox_from_model import return_bbox_from_model

model_file = Path(
    "/home/rafaelfc/Data/DeepFlowImaging/FRCNN/training/simple/best_fRCNN_mask_16.keras"
)
model_config_file = Path(
    "/home/rafaelfc/Data/DeepFlowImaging/FRCNN/training/simple/best_fRCNN_mask_16.keras_CONFIG.h5"
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

out_folder = Path("examples_2")
out_folder.mkdir(exist_ok=True)

img_size = IMG_SIZE

RPN_top_samples = 100000

# Creating the anchors
anchors, index_anchors_valid = create_anchors(
    img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES
)

img_files = os.listdir(
    "/home/rafaelfc/Data/DeepFlowImaging/FRCNN/examples/dataset_FRCNN_PIV_IJMF/Training/images/"
)
for img_file in img_files:
    if not img_file.endswith(".jpg"):
        continue
    img_raw_file = f"/home/rafaelfc/Data/DeepFlowImaging/FRCNN/examples/dataset_FRCNN_PIV_IJMF/Training/images/{img_file}"
    img_mask_file = f"/home/rafaelfc/Data/DeepFlowImaging/FRCNN/examples/dataset_FRCNN_PIV_IJMF/Training/masks/{img_file}"
    __img = cv2.imread(img_raw_file, 0)
    __mask = cv2.imread(img_mask_file, 0)

    __img = __img[: 2 * 512, 72 : (72 + 512)]
    __mask = __mask[: 2 * 512, 72 : (72 + 512)]

    img_raw_rgb_FULL = np.zeros((__img.shape[0], __img.shape[1], 3), dtype=np.uint8)

    # new_img_I = 1100 // 512
    # mega_img = np.zeros(512 * (new_img_I - 1))

    for N in range(2):

        _img = __img[N * 512 : (N + 1) * 512, :512]
        _mask = __mask[N * 512 : (N + 1) * 512, :512]

        # _img = __img[:512, :512]
        # _mask = __mask[:512, :512]

        img_out = _img
        img_mask_rgb = cv2.cvtColor(_mask, cv2.COLOR_GRAY2BGR)
        img_raw_rgb = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
        # # cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        # img_out = imgs[k] * 255.0
        # img_out = img_out.astype(np.uint8)
        # img_mask_rgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
        # img_raw_rgb = img_raws[k] * 255.0
        # img_raw_rgb = img_raw_rgb.astype(np.uint8)
        # img_raw_rgb = cv2.cvtColor(img_raw_rgb, cv2.COLOR_GRAY2BGR)
        #
        #
        # Image in TF/Keras format
        img = np.zeros((1, img_size[0], img_size[1], 1), dtype=np.float64)

        _mask_float = _mask.astype(float)
        _mask_float = _mask / 255.0
        img[0, :, :, 0] = _mask_float
        #
        # Infering the labels(Acc.) and bbox positions from the RPN/f-RCNN model
        inference = model.predict(img)
        labels_pred = inference[0]  # labels - probability of being foreground (Acc.)
        bbox_pred = inference[1]  # bbox leftmost and right most points

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
            BBOX_A = return_bbox_from_model(
                k_A, anchors, bbox_pred_rav, labels_pred_rav
            )

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
            BBOX_A = return_bbox_from_model(
                k_A, anchors, bbox_pred_rav, labels_pred_rav
            )

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
            x_2 = min(x_2, img_size[1])
            y_2 = min(y_2, img_size[0])

            cv2.rectangle(img_mask_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)
            cv2.rectangle(img_raw_rgb, (x_1, y_1), (x_2, y_2), (000, 255, 000), 2)

            # Appending to the bboxes list, rectangles containing the bounding boxes
            bboxes.append((x_1, y_1, x_2, y_2))

        # cv2.imshow("debug", img_mask_rgb)
        # cv2.waitKey(0)
        # cv2.imwrite(os.path.join(out_folder, "fully_%06d.jpg" % N), img_raw_rgb)

        img_raw_rgb_FULL[N * 512 : (N + 1) * 512, :512] = img_raw_rgb

    cv2.imwrite(
        os.path.join(out_folder, f"FULL_{Path(img_file).stem}.jpg"), img_raw_rgb_FULL
    )
