import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
import config as cfg

# Foreground/Background loss function
def loss_cls(y_true, y_pred):
    condition = K.greater(y_true, -0.5)
    indices = tf.where(condition)

    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    loss = K.binary_crossentropy(target, output)
    return K.mean(loss)

# Bounding rectangle regressor loss function
def loss_reg(y_true, y_pred):
    condition = K.less(y_true, 1000.0)
    indices = tf.where(condition)

    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    loss = tf.losses.huber(target, output)
    return cfg.N_RATIO_LOSSES * K.mean(loss)
