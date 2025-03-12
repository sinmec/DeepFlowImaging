from pathlib import Path

import cv2
import numpy as np
from keras.callbacks import Callback

import config as cfg
from FRCNN.training.simple.return_bounding_box_points import return_bounding_box_points
from create_anchors import create_anchors
from return_bbox_from_model import return_bbox_from_model


class TrackProgress(Callback):
    def __init__(self, inputs, out_folder=Path("progress/best")):
        self.out_folder = out_folder
        self.out_folder.mkdir(parents=True, exist_ok=True)

        self.IMG_SIZE = inputs["IMG_SIZE"]
        self.N_SUB = inputs["N_SUB"]
        self.ANCHOR_SIZES = inputs["ANCHOR_SIZES"]
        self.images_verification = inputs["images_verification"]
        self.images_raw_verification = inputs["images_raw_verification"]
        self.bbox_datasets_verification = inputs["bbox_datasets_verification"]

    def on_epoch_end(self, epoch, logs=None):
        EPOCH_INTERVAL = 10
        if not epoch % EPOCH_INTERVAL == 0:
            return

        VAL_DATA_JUMP = 16
        RPN_TOP_SAMPLES = 100000
        SCORE_THRESHOLD = 0.9
        NMS_THRESHOLD = 0.1

        anchors, index_anchors_valid = create_anchors(
            self.IMG_SIZE, self.N_SUB, cfg.ANCHOR_RATIOS, self.ANCHOR_SIZES
        )

        _images = self.images_verification[::VAL_DATA_JUMP].copy()
        _images_raw = self.images_raw_verification[::VAL_DATA_JUMP].copy()
        _bboxes = self.bbox_datasets_verification[::VAL_DATA_JUMP].copy()
        N_PROGRESS_IMGS = _images.shape[0]

        imgs = np.zeros(
            (N_PROGRESS_IMGS, self.IMG_SIZE[0], self.IMG_SIZE[1], 1), dtype=np.float32
        )
        imgs[:, :, :, 0] = _images

        inference = self.model.predict(imgs)

        _images_raw *= 255
        _images_raw = _images_raw.astype(np.uint8)

        _images *= 255
        _images = _images.astype(np.uint8)

        for k in range(N_PROGRESS_IMGS):

            img_rgb = cv2.cvtColor(_images[k], cv2.COLOR_GRAY2BGR)
            img_raw_rgb = cv2.cvtColor(_images_raw[k], cv2.COLOR_GRAY2BGR)

            bbox_dataset = _bboxes[k]
            for _bbox in enumerate(bbox_dataset):
                bbox = _bbox[1]
                if np.isnan(bbox[0]):
                    continue
                p_1, p_2 = return_bounding_box_points(bbox)
                cv2.rectangle(img_rgb, p_1, p_2, (0, 255, 0), 2)
                cv2.rectangle(img_raw_rgb, p_1, p_2, (0, 255, 0), 2)

            bbox_pred = inference[0][k]
            labels_pred = inference[1][k]

            labels_pred_rav = np.ravel(labels_pred)
            bbox_pred_rav = np.ravel(bbox_pred)

            labels_pred_rav_argsort = np.argsort(labels_pred_rav)
            labels_pred_rav_argsort = labels_pred_rav_argsort[-RPN_TOP_SAMPLES:]
            labels_top = labels_pred_rav[labels_pred_rav_argsort]

            bboxes = []
            scores = []
            for m in range(len(labels_top)):
                k_A = labels_pred_rav_argsort[m]
                label_A = labels_pred_rav[k_A]
                BBOX_A = return_bbox_from_model(k_A, anchors, bbox_pred_rav)
                bboxes.append(BBOX_A)
                scores.append(float(label_A))

            nms_indexes = cv2.dnn.NMSBoxes(
                bboxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD
            )

            for index in nms_indexes:
                BBOX_A = bboxes[index]

                x_1 = BBOX_A[0]
                y_1 = BBOX_A[1]
                x_2 = BBOX_A[0] + BBOX_A[2]
                y_2 = BBOX_A[1] + BBOX_A[3]

                x_1 = max(x_1, 0)
                y_1 = max(y_1, 0)
                x_2 = min(x_2, self.IMG_SIZE[1])
                y_2 = min(y_2, self.IMG_SIZE[0])

                cv2.rectangle(img_rgb, (x_1, y_1), (x_2, y_2), (255, 0, 255), 2)
                cv2.rectangle(img_raw_rgb, (x_1, y_1), (x_2, y_2), (255, 0, 255), 2)

            cv2.imwrite(
                str(self.out_folder / f"progress_img_ex_{k:03d}_{epoch:05d}.jpg"),
                np.hstack((img_rgb, img_raw_rgb)),
            )
