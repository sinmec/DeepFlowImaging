import sys
from pathlib import Path


import keras
import keras_tuner
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

N_KT = 8  # 1/N_KT of the dataset is used in 'kt' mode

import cv2
import numpy as np
from read_dataset import read_dataset
from UNET_models import (
    create_econ_model,
    create_mini_model,
    create_large_model,
    create_largest_model,
)


class TrackProgress(Callback):
    def __init__(self, model_name=""):
        self.model_name = model_name
    def on_epoch_end(self, epoch,  logs=None):
        out_progress_folder = Path('progress',f'{self.model_name}')
        if epoch == 0:
            out_progress_folder.mkdir(parents=True, exist_ok=True)
        if epoch % 10 == 0:
            _images = images_val[::16, :, :].copy()
            _masks = masks_val[::16, :, :].copy()

            n_tests = _images.shape[0]
            _pred_masks = model.predict(_images)

            _images *= 255.0
            _masks *= 255.0
            _pred_masks *= 255.0

            for i in range(n_tests):
                _img = _images[i, :, :, 0].astype(np.uint8)
                _mask = _masks[i, :, :, 0].astype(np.uint8)
                _pred_mask = _pred_masks[i, :, :, 0].astype(np.uint8)
                _out_img = np.hstack((_img, _mask, _pred_mask))
                cv2.imwrite(
                    str(
                        Path(
                            out_progress_folder,
                            f"progress_img_ex_{i:03d}_{epoch:05d}.jpg",
                        )
                    ),
                    _out_img,
                )


dataset_folder = Path("/home/rafaelfc/Data/DeepFlowImaging/UNET/examples/dataset_UNET_PIV_IJMF/")
window_size = 128
# run_mode = sys.argv[1]
# unet_base_arch = sys.argv[2]
unet_base_arch = "large"
run_mode = "train"
EPOCHS = 5000
PATIENCE = 500
BATCH_SIZE = 128

if run_mode == "train":
    train_mode = True
elif run_mode == "kt":
    train_mode = False


# I tried to move this 'monster' to a separate .py file, but it is not straight-forward
# as one may seem. It is a bit difficult to pass the hp argument. So... let's keep working
# with the current implementation...
# TODO: Fix it!
def model_builder(hp):
    N_FILTERS = []
    KERNEL_SIZES = []
    DROPOUTS = []

    if unet_base_arch == "mini":
        MAX_POOL_SIZE = 2
        for i in range(4):
            _N_FILTER = hp.Int(
                "n_filter_%02d" % i, min_value=4, max_value=24, default=2, step=4
            )
            N_FILTERS.append(_N_FILTER)
        for i in range(4):
            _KERNEL_SIZE = hp.Int(
                "kernel_size_%02d" % i, min_value=3, max_value=11, default=3, step=2
            )
            KERNEL_SIZES.append(_KERNEL_SIZE)
        for i in range(2):
            _DROPOUT = hp.Float(
                "dropout_%d" % i, min_value=0.0, max_value=0.5, default=0.1, step=0.1
            )
            DROPOUTS.append(_DROPOUT)
        BATCH_MODE = hp.Choice("batch_norm", [True, False])

        model = create_mini_model(
            window_size=window_size,
            n_filters=N_FILTERS,
            dropouts=DROPOUTS,
            kernel_sizes=KERNEL_SIZES,
            max_pool_size=MAX_POOL_SIZE,
            batchnorm=BATCH_MODE,
        )
    elif unet_base_arch == "econ":
        MAX_POOL_SIZE = 2
        for i in range(7):
            _N_FILTER = hp.Int(
                "n_filter_%02d" % i, min_value=4, max_value=24, default=2, step=4
            )
            N_FILTERS.append(_N_FILTER)
        for i in range(7):
            _KERNEL_SIZE = hp.Int(
                "kernel_size_%02d" % i, min_value=3, max_value=11, default=3, step=2
            )
            KERNEL_SIZES.append(_KERNEL_SIZE)
        for i in range(4):
            _DROPOUT = hp.Float(
                "dropout_%d" % i, min_value=0.0, max_value=0.5, default=0.1, step=0.1
            )
            DROPOUTS.append(_DROPOUT)
        BATCH_MODE = hp.Choice("batch_norm", [True, False])

        model = create_econ_model(
            window_size_2=window_size,
            n_filters=N_FILTERS,
            dropouts=DROPOUTS,
            kernel_sizes=KERNEL_SIZES,
            max_pool_size=MAX_POOL_SIZE,
            batchnorm=BATCH_MODE,
        )

    elif unet_base_arch == "large":
        MAX_POOL_SIZE = hp.Choice("max_pool_size", [2, 4])
        # MAX_POOL_SIZE = 2
        for i in range(10):
            _N_FILTER = hp.Int(
                "n_filter_%02d" % i, min_value=4, max_value=24, default=2, step=4
            )
            N_FILTERS.append(_N_FILTER)
        for i in range(10):
            _KERNEL_SIZE = hp.Int(
                "kernel_size_%02d" % i, min_value=3, max_value=11, default=3, step=2
            )
            KERNEL_SIZES.append(_KERNEL_SIZE)
        for i in range(6):
            _DROPOUT = hp.Float(
                "dropout_%d" % i, min_value=0.0, max_value=0.5, default=0.1, step=0.1
            )
            DROPOUTS.append(_DROPOUT)
        BATCH_MODE = hp.Choice("batch_norm", [True, False])

        model = create_large_model(
            window_size=window_size,
            n_filters=N_FILTERS,
            dropouts=DROPOUTS,
            kernel_sizes=KERNEL_SIZES,
            max_pool_size=MAX_POOL_SIZE,
            batchnorm=BATCH_MODE,
        )

    elif unet_base_arch == "largest":
        MAX_POOL_SIZE = hp.Choice("max_pool_size", [2, 4])
        for i in range(13):
            _N_FILTER = hp.Int(
                "n_filter_%02d" % i, min_value=4, max_value=24, default=2, step=4
            )
            N_FILTERS.append(_N_FILTER)
        for i in range(13):
            _KERNEL_SIZE = hp.Int(
                "kernel_size_%02d" % i, min_value=3, max_value=11, default=3, step=2
            )
            KERNEL_SIZES.append(_KERNEL_SIZE)
        for i in range(8):
            _DROPOUT = hp.Float(
                "dropout_%d" % i, min_value=0.0, max_value=0.5, default=0.1, step=0.1
            )
            DROPOUTS.append(_DROPOUT)
        BATCH_MODE = hp.Choice("batch_norm", [True, False])

        model = create_largest_model(
            window_size=window_size,
            n_filters=N_FILTERS,
            dropouts=DROPOUTS,
            kernel_sizes=KERNEL_SIZES,
            max_pool_size=MAX_POOL_SIZE,
            batchnorm=BATCH_MODE,
        )

    model.compile(optimizer=Adam(), loss="binary_crossentropy")

    return model


images_train, masks_train = read_dataset(
    dataset_folder, window_size=window_size, subset="Training"
)
images_val, masks_val = read_dataset(
    dataset_folder, window_size=window_size, subset="Validation"
)

N_dataset = images_train.shape[0]
shuffle = np.arange(N_dataset)
np.random.seed(13)
np.random.shuffle(shuffle)
images_train = images_train[shuffle, :, :, :]
masks_train = masks_train[shuffle, :, :, :]

if not train_mode:
    images_train = images_train[: N_dataset // N_KT, :, :, :]
    masks_train = masks_train[: N_dataset // N_KT, :, :, :]

tuner = keras_tuner.Hyperband(
    model_builder,
    objective="val_loss",
    max_epochs=EPOCHS//5,
    seed=42,
    directory=f"hyperband_{unet_base_arch}",
    project_name=f"unet_run_{unet_base_arch}",
)

stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
# modificando na marra o 'self.min_epochs = 100' no hyperband.py...
tuner.search(
    x=images_train,
    y=masks_train,
    validation_data=(images_val, masks_val),
    epochs=20,
    initial_epoch=200,
    callbacks=[stop_early],
    batch_size=BATCH_SIZE
)

if train_mode:

    best_hps = tuner.get_best_hyperparameters(num_trials=10)

    for trial, best_hp in enumerate(best_hps):
        model = tuner.hypermodel.build(best_hp)
        model_name = f"model_{unet_base_arch}_{trial:02d}"

        checkpoint = ModelCheckpoint(
            f"UNET_best_{model_name}.keras",
            verbose=1,
            save_best_only=True,
            monitor="val_loss",
            mode="auto",
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=PATIENCE
        )

        model.fit(
            images_train,
            masks_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(images_val, masks_val),
            callbacks=[checkpoint, early_stopping, TrackProgress(model_name=model_name)],
        )
