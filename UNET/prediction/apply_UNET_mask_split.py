import numpy as np


def apply_UNET_mask(subdivided_image, model):
    subdivided_image = subdivided_image.astype(np.float64) / 255.0
    subdivided_UNET_image = model.predict(subdivided_image, verbose = 0)
    subdivided_UNET_image *= 255
    subdivided_UNET_image = subdivided_UNET_image.astype(np.uint8)

    return subdivided_UNET_image
