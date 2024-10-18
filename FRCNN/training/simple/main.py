from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
import sys
import config as cfg
from input_generator import input_generator
from losses import loss_cls, loss_reg
from read_dataset import read_dataset
from save_model_configuration import save_model_configuration
import cv2
from create_anchors import create_anchors
from return_bbox_from_model import return_bbox_from_model

img_size = cfg.IMG_SIZE

N_SUB = 16
ANCHOR_SIZES = np.array(cfg.ANCHOR_REAL_SIZE) // N_SUB
N_ANCHORS = len(ANCHOR_SIZES)
N_RATIOS = len(cfg.ANCHOR_RATIOS)

MODEL_OPTIONS = {
    "N_SUB": N_SUB,
    "ANCHOR_SIZES": ANCHOR_SIZES,
    "N_ANCHORS": N_ANCHORS,
    "N_RATIOS": N_RATIOS,
}


best_model_name = f"best_fRCNN_{cfg.MODE}_{N_SUB:02d}.keras"
filepath = Path(f"{Path(best_model_name).stem}_CONFIG.h5")
save_model_configuration(filepath, N_SUB, ANCHOR_SIZES)

dataset_folder = Path(
    "../../examples/example_dataset_FRCNN_PIV_subimage/Output"
)
images_train, bbox_datasets_train, _ = read_dataset(
    img_size, dataset_folder, subset="Training"
)
images_val, bbox_datasets_val, raw_val_images = read_dataset(
    img_size, dataset_folder, subset="Validation"
)

input_image = Input(shape=(img_size[0], img_size[1], 1))
conv_3_3_1 = Conv2D(
    filters=cfg.N_FILTERS, kernel_size=cfg.KERNEL_SIZE, padding="same", name="3x3-1"
)(input_image)
max_pool_1 = MaxPooling2D((2, 2), name="max_pool_1")(conv_3_3_1)

conv_3_3_2 = Conv2D(
    filters=cfg.N_FILTERS, kernel_size=cfg.KERNEL_SIZE, padding="same", name="3x3-2"
)(max_pool_1)

max_pool_2 = MaxPooling2D((2, 2), name="max_pool_2")(conv_3_3_2)

conv_3_3_3 = Conv2D(
    filters=cfg.N_FILTERS, kernel_size=cfg.KERNEL_SIZE, padding="same", name="3x3-3"
)(max_pool_2)

max_pool_3 = MaxPooling2D((2, 2), name="max_pool_3")(conv_3_3_3)

conv_3_3_4 = Conv2D(
    filters=cfg.N_FILTERS, kernel_size=cfg.KERNEL_SIZE, padding="same", name="3x3-4"
)(max_pool_3)

max_pool_4 = MaxPooling2D((2, 2), name="max_pool_4")(conv_3_3_4)

conv_3_3_5 = Conv2D(
    filters=cfg.N_FILTERS, kernel_size=cfg.KERNEL_SIZE, padding="same", name="3x3-5"
)(max_pool_4)

max_pool_5 = MaxPooling2D((2, 2), name="max_pool_5")(conv_3_3_5)

conv_3_3_6 = Conv2D(
    filters=cfg.N_FILTERS, kernel_size=cfg.KERNEL_SIZE, padding="same", name="3x3-6"
)(max_pool_5)

max_pool_6 = MaxPooling2D((2, 2), name="max_pool_6")(conv_3_3_6)

if N_SUB == 1:
    last_layer = conv_3_3_1
elif N_SUB == 2:
    last_layer = max_pool_1
elif N_SUB == 4:
    last_layer = max_pool_2
elif N_SUB == 8:
    last_layer = max_pool_3
elif N_SUB == 16:
    last_layer = max_pool_4
elif N_SUB == 32:
    last_layer = max_pool_5

output_scores = Conv2D(
    filters=N_ANCHORS * N_RATIOS,
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="uniform",
    name="l_reg",
)(last_layer)

output_regressor = Conv2D(
    filters=N_ANCHORS * N_RATIOS * 4,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="bb_reg",
)(last_layer)

opt = Adam(learning_rate=cfg.ADAM_LEARNING_RATE)
model = Model(inputs=[input_image], outputs=[output_regressor, output_scores])
model.compile(optimizer=opt, loss={"l_reg": loss_cls, "bb_reg": loss_reg})

plot_model(model, show_shapes=True, to_file="model_true.png")
model.summary()

checkpoint = ModelCheckpoint(
    best_model_name, verbose=1, save_best_only=True, monitor="val_loss", mode="auto"
)

early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=cfg.N_PATIENCE
)

class TrackProgress(Callback):
    def __init__(self, out_folder="progress/best"):
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            anchors, index_anchors_valid = create_anchors(
                img_size, N_SUB, cfg.ANCHOR_RATIOS, ANCHOR_SIZES
            )

            _images = raw_val_images[::16].copy()
            _bboxes = bbox_datasets_val[::16]
            n_tests = _images.shape[0]
            RPN_top_samples = 100000

            print(f"NUMERO DE IMAGENS {n_tests}")

            for k in range(n_tests):
                img_out = _images[k].copy() * 255.0
                img_out = img_out.astype(np.uint8)
                img_mask_rgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

                img_raw_rgb = _images[k].copy() * 255.0
                img_raw_rgb = img_raw_rgb.astype(np.uint8)
                img_raw_rgb = cv2.cvtColor(img_raw_rgb, cv2.COLOR_GRAY2BGR)

                bbox_dataset = _bboxes[k]
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
                    # cv2.rectangle(img_mask_rgb, p_1, p_2, (0, 255, 255), 2)
                    cv2.rectangle(img_raw_rgb, p_1, p_2, (255, 0, 0), 2)

                img = np.zeros((1, img_size[0], img_size[1], 1), dtype=np.float64)
                img[0, :, :, 0] = _images[k]

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

                    cv2.rectangle(img_mask_rgb, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

                    bboxes.append((x_1, y_1, x_2, y_2))

                cv2.imwrite(
                    str(self.out_folder / f"progress_img_ex_{k:03d}_{epoch:05d}.jpg"),
                    np.hstack((img_raw_rgb, img_mask_rgb)),
                )


model.fit(
    input_generator(images_train, bbox_datasets_train, MODEL_OPTIONS),
    validation_data=input_generator(images_val, bbox_datasets_val, MODEL_OPTIONS),
    validation_steps=100,
    steps_per_epoch=100,
    epochs=cfg.N_EPOCHS,
    callbacks=[checkpoint, early_stopping, TrackProgress()],
)
