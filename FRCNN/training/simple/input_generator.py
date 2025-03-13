import numpy as np

import config as cfg
from calculate_bbox_intesect_over_union import calculate_bbox_intesect_over_union
from create_samples_for_training import create_samples_for_training
from evaluate_ious import evaluate_ious
from parametrize_anchor_box_properties import parametrize_anchor_box_properties


def input_generator(imgs, bbox_datasets, model_options):
    N_SUB = model_options["N_SUB"]
    N_ANCHORS = model_options["N_ANCHORS"]
    N_RATIOS = model_options["N_RATIOS"]
    img_size = cfg.IMG_SIZE

    anchors = model_options["anchors"]
    index_anchors_valid = model_options["index_anchors_valid"]

    while 1:
        random_indexes = np.random.randint(
            low=0, high=len(imgs) - 1, size=cfg.BATCH_SIZE_IMAGES
        )
        batch_imgs = np.zeros(
            (len(random_indexes), img_size[0], img_size[1], 1), dtype=np.float32
        )
        batch_anchor_labels = np.zeros(
            (
                len(random_indexes),
                img_size[0] // N_SUB,
                img_size[1] // N_SUB,
                N_ANCHORS * N_RATIOS,
            ),
            dtype=np.float32,
        )
        batch_anchor_locations = np.zeros(
            (
                len(random_indexes),
                img_size[0] // N_SUB,
                img_size[1] // N_SUB,
                4 * N_ANCHORS * N_RATIOS,
            ),
            dtype=np.float32,
        )

        for k, random_index in enumerate(random_indexes):
            img = imgs[random_index]
            bbox_dataset = bbox_datasets[random_index]

            ious = calculate_bbox_intesect_over_union(
                anchors, index_anchors_valid, bbox_dataset, img
            )

            labels, anchor_argmax_ious = evaluate_ious(
                anchors,
                index_anchors_valid,
                ious,
                bbox_dataset,
                img,
                cfg.POS_IOU_THRESHOLD,
                cfg.NEG_IOU_THRESHOLD,
            )

            anchor_labels = create_samples_for_training(
                anchors,
                labels,
                bbox_dataset,
                img,
                debug=False,
            )

            anchor_labels = np.reshape(
                anchor_labels,
                (img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS),
            )

            anchor_locations = parametrize_anchor_box_properties(
                anchors, anchor_argmax_ious, labels, ious, bbox_dataset, img
            )

            anchor_locations = np.reshape(
                anchor_locations,
                (img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS),
            )

            anchor_labels = anchor_labels.astype(np.float32)

            batch_imgs[k, :, :, 0] = img

            batch_anchor_labels[k, :, :, :] = anchor_labels
            batch_anchor_locations[k, :, :, :] = anchor_locations

        yield batch_imgs, (batch_anchor_locations, batch_anchor_labels)
