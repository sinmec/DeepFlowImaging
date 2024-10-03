from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

import config as cfg
from input_generator import input_generator
from losses import loss_cls, loss_reg
from read_dataset import read_dataset
from save_model_configuration import save_model_configuration

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
    "/home/rafaelfc/Data/DATASETS/example_dataset_FRCNN_PIV_subimage/Output"
)
images_train, bbox_datasets_train, _ = read_dataset(
    img_size, dataset_folder, subset="Training"
)
images_val, bbox_datasets_val, _ = read_dataset(
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

model.fit(
    input_generator(images_train, bbox_datasets_train, MODEL_OPTIONS),
    validation_data=input_generator(images_val, bbox_datasets_val, MODEL_OPTIONS),
    validation_steps=100,
    steps_per_epoch=100,
    epochs=cfg.N_EPOCHS,
    callbacks=[checkpoint, early_stopping],
)
