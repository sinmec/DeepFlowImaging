import concurrent.futures
import os
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from UNET.prediction.apply_UNET_mask_split import apply_UNET_mask
from UNET.prediction.divide_image import divide_image
from UNET.prediction.recreate_UNET_image import recreate_UNET_image

image_file_folder = r"/EXPERIMENTS/HSC_images/22_08_Ql_1_P_1_HIGH_20__001"

UNET_model_file = os.path.join("../training/simple/UNET_best.keras")
UNET_model = keras.models.load_model(UNET_model_file, compile=False)
window_size = 128
sub_image_size = 64
stride_division = 8

out_folder = Path("OUTPUT")
out_folder.mkdir(parents=True, exist_ok=True)

image_files = []
for _file in os.listdir(image_file_folder):
    if _file.endswith(".jpg"):
        image_files.append(_file)
image_files.sort()


def process_image(index_img, image_file):
    img = cv2.imread(str(Path(image_file_folder, image_file)), 0)
    subdivided_image_raw = divide_image(
        window_size, sub_image_size, img, stride_division
    )

    subdivided_image_UNET = apply_UNET_mask(subdivided_image_raw, UNET_model)
    img_UNET = recreate_UNET_image(
        subdivided_image_UNET, window_size, img, stride_division
    )

    out_img = np.hstack((img, img_UNET))
    cv2.imwrite(str(Path(out_folder, f"img_{index_img:06d}.jpg")), out_img)


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for index_img, image_file in enumerate(image_files):
        future = executor.submit(process_image, index_img, image_file)
        futures.append(future)

    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Processing images",
    ):
        future.result()
