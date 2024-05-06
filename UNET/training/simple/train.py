from pathlib import Path

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam

from conv2d_block import conv2d_block
from read_dataset import read_dataset


class TrackProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        out_progress_folder = Path("progress/best")
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


dataset_folder = Path("../../dataset/dataset_UNET")
window_size = 64
EPOCHS = 5000
PATIENCE = 500
BATCH_SIZE = 128

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

input_img = Input((window_size, window_size, 1), name="img")

c1 = conv2d_block(input_img, 8, kernel_size=5, batchnorm=True)
p1 = MaxPooling2D((4, 4))(c1)

c2 = conv2d_block(p1, 8, kernel_size=5, batchnorm=True)
p2 = MaxPooling2D((4, 4))(c2)

c3 = conv2d_block(p2, 8, kernel_size=5, batchnorm=True)

u4 = Conv2DTranspose(8, kernel_size=5, strides=(4, 4), padding="same")(c3)
u4 = concatenate([u4, c2])
c4 = conv2d_block(u4, 8, kernel_size=5, batchnorm=True)

u5 = Conv2DTranspose(8, kernel_size=5, strides=(4, 4), padding="same")(c4)
u5 = concatenate([u5, c1])
c5 = conv2d_block(u5, 8, kernel_size=5, batchnorm=True)

outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)
model = Model(inputs=[input_img], outputs=[outputs])
model.compile(optimizer=Adam(), loss="binary_crossentropy")

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=PATIENCE
)

checkpoint = ModelCheckpoint(
    "UNET_best.keras", verbose=1, save_best_only=True, monitor="val_loss", mode="auto"
)

model.fit(
    images_train,
    masks_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(images_val, masks_val),
    callbacks=[checkpoint, early_stopping, TrackProgress()],
)
