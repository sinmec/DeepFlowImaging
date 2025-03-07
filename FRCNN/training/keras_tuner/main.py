import math
from pathlib import Path

import keras_tuner as kt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

import config as cfg
from FRCNN.training.keras_tuner.callbacks import TrackProgress
from input_generator import input_generator
from losses import loss_cls, loss_reg
from read_dataset import read_dataset
from save_model_configuration import save_model_configuration

KT_MODE = "RandomSearch"
# KT_MODE = "HyperBand"

IMG_SIZE = cfg.IMG_SIZE

N_SUB = 8

if N_SUB == 32:
    ANCHOR_REAL_SIZE = [32, 64, 96]
    POS_IOU_THRESHOLD = 0.35
    NEG_IOU_THRESHOLD = 0.1
elif N_SUB == 16:
    ANCHOR_REAL_SIZE = [16, 32, 48, 64, 96]
    POS_IOU_THRESHOLD = 0.50
    NEG_IOU_THRESHOLD = 0.1
elif N_SUB == 8:
    ANCHOR_REAL_SIZE = [8, 16, 32, 48, 64, 96]
    POS_IOU_THRESHOLD = 0.60
    NEG_IOU_THRESHOLD = 0.1

ANCHOR_SIZES = np.array(ANCHOR_REAL_SIZE) // N_SUB
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

dataset_folder = Path(cfg.INPUT_FOLDER)
images_train, bbox_datasets_train, _ = read_dataset(
    IMG_SIZE, dataset_folder, mode=cfg.MODE, subset="Training"
)
images_val, bbox_datasets_val, _ = read_dataset(
    IMG_SIZE, dataset_folder, mode=cfg.MODE, subset="Validation"
)
images_verification, bbox_datasets_verification, images_raw_verification = read_dataset(
    IMG_SIZE, dataset_folder, mode=cfg.MODE, subset="Verification"
)


def model_builder(hp):
    input_image = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))

    N_conv_layers = int(math.log2(N_SUB))

    for i in range(N_conv_layers):
        _N_FILTERS = hp.Int(
            "n_filters_%02d" % i, min_value=3, max_value=21, default=7, step=3
        )
        _N_KERNELS = hp.Int(
            "n_kernels_%02d" % i, min_value=5, max_value=7, default=5, step=2
        )
        if i == 0:
            _conv = Conv2D(
                filters=_N_FILTERS,
                kernel_size=_N_KERNELS,
                padding="same",
                name="conv_%02d" % i,
            )(input_image)
        else:
            _conv = Conv2D(
                filters=_N_FILTERS,
                kernel_size=_N_KERNELS,
                padding="same",
                name="conv_%02d" % i,
            )(_max_pool)
        _max_pool = MaxPooling2D((2, 2), name="max_pool_%02d" % i)(_conv)

    output_scores = Conv2D(
        filters=N_ANCHORS * N_RATIOS,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="l_reg",
    )(_max_pool)

    output_regressor = Conv2D(
        filters=N_ANCHORS * N_RATIOS * 4,
        kernel_size=(1, 1),
        activation="linear",
        kernel_initializer="uniform",
        name="bb_reg",
    )(_max_pool)

    opt = Adam(learning_rate=cfg.ADAM_LEARNING_RATE)
    model = Model(inputs=[input_image], outputs=[output_regressor, output_scores])
    model.compile(optimizer=opt, loss={"l_reg": loss_cls, "bb_reg": loss_reg})

    plot_model(model, show_shapes=True, to_file="model.png")

    return model


project_name = f"{KT_MODE}_{cfg.MODE}_{N_SUB}_{cfg.N_RATIO_LOSSES:.1f}"
if KT_MODE == "HyperBand":
    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        max_epochs=500,
        # min_epochs=50, # estou mudando direto no source code
        factor=3,
        seed=42,
        project_name=project_name,
    )

elif KT_MODE == "RandomSearch":
    tuner = kt.RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=64,
        seed=42,
        project_name=project_name,
    )

early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)

tuner.search(
    input_generator(images_train, bbox_datasets_train, MODEL_OPTIONS),
    validation_data=input_generator(images_val, bbox_datasets_val, MODEL_OPTIONS),
    validation_steps=20,
    steps_per_epoch=100,
    epochs=200,
    callbacks=[early_stopping],
)

best_hps = tuner.get_best_hyperparameters(num_trials=5)
for trial, best_hp in enumerate(best_hps):
    best_model_name = (
        f"best_fRCNN_{cfg.MODE}_{N_SUB:02d}_{cfg.N_RATIO_LOSSES:.1f}_{trial:02d}"
    )
    filepath = Path(f"{Path(best_model_name).stem}_CONFIG.h5")
    save_model_configuration(filepath, N_SUB, ANCHOR_SIZES)

    model = tuner.hypermodel.build(best_hp)

    checkpoint = ModelCheckpoint(
        Path(f"{best_model_name}.keras"),
        verbose=1,
        save_best_only=True,
        monitor="val_loss",
        mode="auto",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=cfg.N_PATIENCE
    )

    calllback_progress_inputs = {
        "IMG_SIZE": IMG_SIZE,
        "N_SUB": N_SUB,
        "ANCHOR_SIZES": ANCHOR_SIZES,
        "images_verification": images_verification,
        "images_raw_verification": images_raw_verification,
        "bbox_datasets_verification": bbox_datasets_verification,
    }

    model.fit(
        input_generator(images_train, bbox_datasets_train, MODEL_OPTIONS),
        validation_data=input_generator(images_val, bbox_datasets_val, MODEL_OPTIONS),
        validation_steps=100,
        steps_per_epoch=100,
        epochs=cfg.N_EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping,
            TrackProgress(
                calllback_progress_inputs, out_folder=Path("progress", best_model_name)
            ),
        ],
    )
