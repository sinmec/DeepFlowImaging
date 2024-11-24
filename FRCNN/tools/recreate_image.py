import numpy as np

def recreate_image(imgs, original_size, pos):
    recreated_img = np.zeros([original_size[0], original_size[1], 3], dtype=np.uint8)

    for img, (y1, y2, x1, x2) in zip(imgs, pos):
        recreated_img[y1:y2, x1:x2] = img

    return recreated_img
