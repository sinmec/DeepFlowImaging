
from pathlib import Path

import cv2
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
from tensorflow import keras
import random
import h5py as h5
import math

import keras_tuner as kt

import config as cfg
from FRCNN.training.simple.save_model_configuration import save_model_configuration

from read_dataset import read_dataset
from create_anchors import create_anchors
from calculate_bbox_intesect_over_union import calculate_bbox_intesect_over_union
from evaluate_ious import evaluate_ious
from create_samples_for_training import create_samples_for_training
from parametrize_anchor_box_properties import parametrize_anchor_box_properties

from losses import loss_cls, loss_reg
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


# Loading image size from config file
img_size = cfg.IMG_SIZE

# Subscaling/Anchor values (2^n), ex: 1, 2, 4, 8, 16, 32
N_SUB = 16

# Defining anchor sizes
ANCHOR_SIZES = np.array(cfg.ANCHOR_REAL_SIZE) // N_SUB

# Defininf number of anchors sized and rations
N_ANCHORS = len(ANCHOR_SIZES)
N_RATIOS = len(cfg.ANCHOR_RATIOS)


# Defining the best fRCNN model name
best_model_name = f"best_fRCNN_{cfg.MODE}_{N_SUB:02d}.keras"
filepath = Path(f'{Path(best_model_name).stem}_CONFIG.h5')
save_model_configuration(filepath, N_SUB, ANCHOR_SIZES)

# Defining the dataset folder
dataset_folder = Path("/home/rafaelfc/Data/DeepFlowImaging/FRCNN/examples/example_dataset_FRCNN_PIV_subimage/Output/")
images_train, bbox_datasets_train, _ = read_dataset(img_size, dataset_folder, subset="Training")
images_val, bbox_datasets_val, _ = read_dataset(img_size, dataset_folder, subset="Validation")

# Building model
# TODO: Meter keras-tuner aqui
input_image = Input(shape=(img_size[0], img_size[1], 1))
conv_3_3_1 = Conv2D(
    filters=cfg.N_FILTERS,
    kernel_size=cfg.KERNEL_SIZE,
    padding='same',
    name="3x3-1"
)(input_image)
max_pool_1 = MaxPooling2D((2, 2),
                          name="max_pool_1")(conv_3_3_1)

conv_3_3_2 = Conv2D(
    filters=cfg.N_FILTERS,
    kernel_size=cfg.KERNEL_SIZE,
    padding='same',
    name="3x3-2"
)(max_pool_1)

max_pool_2 = MaxPooling2D((2, 2),
                          name="max_pool_2")(conv_3_3_2)

conv_3_3_3 = Conv2D(
    filters=cfg.N_FILTERS,
    kernel_size=cfg.KERNEL_SIZE,
    padding='same',
    name="3x3-3"
)(max_pool_2)

max_pool_3 = MaxPooling2D((2, 2),
                          name="max_pool_3")(conv_3_3_3)

conv_3_3_4 = Conv2D(
    filters=cfg.N_FILTERS,
    kernel_size=cfg.KERNEL_SIZE,
    padding='same',
    name="3x3-4"
)(max_pool_3)

max_pool_4 = MaxPooling2D((2, 2),
                          name="max_pool_4")(conv_3_3_4)

conv_3_3_5 = Conv2D(
    filters=cfg.N_FILTERS,
    kernel_size=cfg.KERNEL_SIZE,
    padding='same',
    name="3x3-5"
)(max_pool_4)

max_pool_5 = MaxPooling2D((2, 2),
                          name="max_pool_5")(conv_3_3_5)

conv_3_3_6 = Conv2D(
    filters=cfg.N_FILTERS,
    kernel_size=cfg.KERNEL_SIZE,
    padding='same',
    name="3x3-6"
)(max_pool_5)

max_pool_6 = MaxPooling2D((2, 2),
                          name="max_pool_6")(conv_3_3_6)

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
    name="l_reg"
)(last_layer)

output_regressor = Conv2D(
    filters=N_ANCHORS * N_RATIOS * 4,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="bb_reg"
)(last_layer)

opt = tf.keras.optimizers.Adam(learning_rate=cfg.ADAM_LEARNING_RATE)
model = Model(inputs=[input_image], outputs=[output_regressor, output_scores])
model.compile(optimizer=opt, loss={'l_reg': loss_cls, 'bb_reg': loss_reg})

plot_model(model, show_shapes=True, to_file="model_true.png")
model.summary()

def input_generator(imgs, bbox_datasets):
    # Creating the anchors
    anchors, index_anchors_valid = create_anchors(img_size, N_SUB, cfg.ANCHOR_RATIOS, ANCHOR_SIZES)

    while 1:
        # Picking a random number of images for training
        random_indexes = np.random.randint(low=0, high=len(imgs) - 1, size=cfg.N_DATA_EPOCHS)
        # Initializing the arrays
        batch_imgs = np.zeros((len(random_indexes), img_size[0], img_size[1], 1), dtype=np.float64)
        batch_anchor_labels = np.zeros(
            (len(random_indexes), img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS), dtype=np.float64)
        batch_anchor_locations = np.zeros(
            (len(random_indexes), img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS),
            dtype=np.float64)

        # print('input', random_indexes, len(random_indexes))

        # Looping over the selected indexes and generating the input dataset
        for k, random_index in enumerate(random_indexes):
            # Retriegving the image and the bbox-values
            img = imgs[random_index]
            bbox_dataset = bbox_datasets[random_index]

            # Calculating anchor/bbox_dataset IoUs
            ious = calculate_bbox_intesect_over_union(anchors, index_anchors_valid, bbox_dataset, img)

            # Evaluating if the anchors are valid or invalid based on the IoUs
            labels, anchor_argmax_ious = evaluate_ious(anchors, index_anchors_valid, ious, bbox_dataset, img,
                                                       cfg.POS_IOU_THRESHOLD, cfg.NEG_IOU_THRESHOLD)

            # Creating the samples for training
            # if k == 1:
            # anchor_labels = create_samples_for_training(anchors, index_anchors_valid, anchor_argmax_ious, labels, ious, bbox_dataset, img, debug=False)
            # else:
            # anchor_labels = create_samples_for_training(anchors, index_anchors_valid, anchor_argmax_ious, labels, ious, bbox_dataset, img, debug=False)
            anchor_labels = create_samples_for_training(anchors, index_anchors_valid, anchor_argmax_ious, labels, ious,
                                                        bbox_dataset, img, debug=False)

            # Reshaping the anchor labels to follow the image/sub-image coordinates
            anchor_labels = np.reshape(anchor_labels,
                                       (img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS))

            # Parametrizing the anchor box properties
            anchor_locations = parametrize_anchor_box_properties(anchors, anchor_argmax_ious, labels, ious,
                                                                 bbox_dataset, img)

            # Reshaping the anchor locations to follow the image/sub-image coordinates
            anchor_locations = np.reshape(anchor_locations,
                                          (img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS))

            # Converting to float
            anchor_labels = anchor_labels.astype(np.float64)

            # Storing images
            batch_imgs[k, :, :, 0] = img

            # print('input', random_indexes)

            # Updating anchor labels and properties(locations)
            batch_anchor_labels[k, :, :, :] = anchor_labels
            batch_anchor_locations[k, :, :, :] = anchor_locations

        # Returning/Yielding samples for model training
        yield batch_imgs, (batch_anchor_locations, batch_anchor_labels)


# Model checkpoint for saving best models
checkpoint = ModelCheckpoint(best_model_name,
                             verbose=1,
                             save_best_only=True,
                             monitor='val_loss',
                             mode='auto')

# Model checkpoint for early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               mode='min',
                               verbose=1,
                               patience=cfg.N_PATIENCE)




model.fit(input_generator(images_train, bbox_datasets_train),
          validation_data=input_generator(images_val, bbox_datasets_val),
          validation_steps=100,
          steps_per_epoch=100,
          epochs=cfg.N_EPOCHS,
          callbacks=[checkpoint, early_stopping])



print('Done!')
