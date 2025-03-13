import numpy as np
from numba import njit

import config as cfg
from FRCNN.training.simple.create_anchors import create_anchors
from calculate_bbox_intesect_over_union import calculate_bbox_intesect_over_union
from create_samples_for_training import create_samples_for_training
from evaluate_ious import evaluate_ious
from parametrize_anchor_box_properties import parametrize_anchor_box_properties


@njit
def process_batch(
    imgs,
    bbox_datasets,
    anchors,
    index_anchors_valid,
    N_SUB,
    img_size,
    N_ANCHORS,
    N_RATIOS,
):
    batch_size = len(imgs)

    batch_imgs = np.zeros((batch_size, img_size[0], img_size[1], 1), dtype=np.float32)
    batch_anchor_labels = np.zeros(
        (batch_size, img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS),
        dtype=np.float32,
    )
    batch_anchor_locations = np.zeros(
        (
            batch_size,
            img_size[0] // N_SUB,
            img_size[1] // N_SUB,
            4 * N_ANCHORS * N_RATIOS,
        ),
        dtype=np.float32,
    )

    for k in range(batch_size):
        img = imgs[k]
        bbox_dataset = bbox_datasets[k]

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

        batch_imgs[k, :, :, 0] = img
        batch_anchor_labels[k, :, :, :] = anchor_labels
        batch_anchor_locations[k, :, :, :] = anchor_locations

    return batch_imgs, batch_anchor_locations, batch_anchor_labels


def dataset_generator(imgs, bbox_datasets_np):

    N_SUB = cfg.N_SUB
    ANCHOR_SIZES = np.array(cfg.ANCHOR_REAL_SIZE) // N_SUB
    N_ANCHORS = len(ANCHOR_SIZES)
    N_RATIOS = len(cfg.ANCHOR_RATIOS)
    IMG_SIZE = cfg.IMG_SIZE

    anchors, index_anchors_valid = create_anchors(
        IMG_SIZE, N_SUB, cfg.ANCHOR_RATIOS, ANCHOR_SIZES
    )

    bbox_datasets = []
    for k in range(imgs.shape[0]):
        _bbox_dataset = bbox_datasets_np[k]
        for j in range(len(_bbox_dataset)):
            _bbox = _bbox_dataset[j].copy()
            if np.isnan(_bbox[0]):
                index_to_remove = j
                break
        bbox_datasets.append(_bbox_dataset[:index_to_remove])

    indices = np.arange(len(imgs))


    while True:
        np.random.shuffle(indices)

        for i in range(0, len(indices), cfg.N_DATA_EPOCHS):
            batch_indexes = indices[i: i + cfg.N_DATA_EPOCHS]

            batch_imgs = [imgs[i] for i in batch_indexes]
            batch_bbox_datasets = [bbox_datasets[i] for i in batch_indexes]

            batch_imgs, batch_anchor_locations, batch_anchor_labels = process_batch(
                batch_imgs,
                batch_bbox_datasets,
                anchors,
                index_anchors_valid,
                N_SUB,
                cfg.IMG_SIZE,
                N_ANCHORS,
                N_RATIOS,
            )
            yield np.array(batch_imgs, dtype=np.float32), (
                np.array(batch_anchor_locations, dtype=np.float32),
                np.array(batch_anchor_labels, dtype=np.float32),
            )
