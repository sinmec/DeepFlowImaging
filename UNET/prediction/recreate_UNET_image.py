import cv2
import numpy as np


# import config as cfg

def recreate_UNET_image(subdivided_image_UNET, window_size, raw_img, stride_division=2):

    UNET_image = np.zeros_like(raw_img, dtype=float)
    UNET_image_nmax = np.zeros_like(raw_img, dtype=float)

    N_I = raw_img.shape[0] // window_size + 1
    N_J = raw_img.shape[1] // window_size + 1

    diff_I = - raw_img.shape[0] + N_I * window_size
    diff_J = - raw_img.shape[1] + N_J * window_size
    assert diff_I % 2 == 0, "diff_I is not even!"
    assert diff_J % 2 == 0, "diff_J is not even!"

    scan_I = np.arange(0, raw_img.shape[0] - window_size + 1, window_size // stride_division)
    scan_J = np.arange(0, raw_img.shape[1] - window_size + 1, window_size // stride_division)


    # Generating the U-Net image 
    index = 0
    for i in scan_I:
        for j in scan_J:
            i_start = i
            i_end = i + window_size
            j_start = j
            j_end = j + window_size

            # Returning the 'intermediary' U-Net from the sliding window full matrix
            # See that I am summing this 'sub-image' into the original/desired U-Net image
            # That is the 'U-Net numerator'
            sub_window = cv2.resize(subdivided_image_UNET[index, :, :, 0], (window_size,window_size))
            UNET_image[i_start:i_end, j_start:j_end] += sub_window
            # UNET_image[i_start:i_end, j_start:j_end] = subdivided_image_UNET[index, :, :, 0]

            # Since we are summing in the numerator, and the values are bounded to 0 and 255
            # I have a second matrix, the UNET_image_nmax, which is a denominator
            # Therefore, I am updating the denominator here
            UNET_image_nmax[i_start:i_end, j_start:j_end] += 255.0
            index += 1

    # print("PROBLEM WHILE GENERATING U-NET IMAGES. FIX IT!!! USE 1 STRIDE_DIV")
    #    # Now I divide the image
    UNET_image /= UNET_image_nmax
    UNET_image /= np.nanmax(UNET_image)
    UNET_image *= 255.0
    UNET_image = UNET_image.astype(np.uint8)

    return  UNET_image
