import numpy as np
from numba import njit

import config as cfg
from calculate_bbox_intesect_over_union import calculate_bbox_intesect_over_union
from create_samples_for_training import create_samples_for_training
from evaluate_ious import evaluate_ious
from parametrize_anchor_box_properties import parametrize_anchor_box_properties


@njit
def process_batch(imgs, bbox_datasets, anchors, index_anchors_valid, N_SUB, img_size, N_ANCHORS, N_RATIOS,
                  POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD):
    batch_size = len(imgs)

    batch_imgs = np.zeros((batch_size, img_size[0], img_size[1], 1), dtype=np.float32)
    batch_anchor_labels = np.zeros((batch_size, img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS),
                                   dtype=np.float32)
    batch_anchor_locations = np.zeros(
        (batch_size, img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS), dtype=np.float32)

    for k in range(batch_size):
        img = imgs[k]
        bbox_dataset = bbox_datasets[k]

        ious = calculate_bbox_intesect_over_union(anchors, index_anchors_valid, bbox_dataset, img)

        labels, anchor_argmax_ious = evaluate_ious(anchors,
                index_anchors_valid,
                ious,
                bbox_dataset,
                img,
                cfg.POS_IOU_THRESHOLD,
                cfg.NEG_IOU_THRESHOLD)

        anchor_labels = create_samples_for_training( anchors,
                labels,
                bbox_dataset,
                img,
                debug=False,)

        anchor_labels = np.reshape(anchor_labels, (img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS))

        anchor_locations = parametrize_anchor_box_properties(anchors, anchor_argmax_ious, labels, ious, bbox_dataset,
                                                             img)

        anchor_locations = np.reshape(anchor_locations,
                                      (img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS))

        batch_imgs[k, :, :, 0] = img
        batch_anchor_labels[k, :, :, :] = anchor_labels
        batch_anchor_locations[k, :, :, :] = anchor_locations

    return batch_imgs, batch_anchor_locations, batch_anchor_labels

def dataset_generator(imgs, bbox_datasets, model_options):
    N_SUB = model_options["N_SUB"]
    N_ANCHORS = model_options["N_ANCHORS"]
    N_RATIOS = model_options["N_RATIOS"]
    img_size = cfg.IMG_SIZE

    anchors = model_options['anchors']
    index_anchors_valid = model_options['index_anchors_valid']

    while True:
        random_indexes = np.random.randint(0, len(imgs), size=cfg.N_DATA_EPOCHS)
        batch_imgs = [imgs[i] for i in random_indexes]
        batch_bbox_datasets = [bbox_datasets[i] for i in random_indexes]

        batch_imgs, batch_anchor_locations, batch_anchor_labels = process_batch(
            batch_imgs, batch_bbox_datasets, anchors, index_anchors_valid,
            model_options["N_SUB"], cfg.IMG_SIZE, model_options["N_ANCHORS"],
            model_options["N_RATIOS"], cfg.POS_IOU_THRESHOLD, cfg.NEG_IOU_THRESHOLD
        )

        # yield batch_imgs, (batch_anchor_locations, batch_anchor_labels)
        yield np.array(batch_imgs, dtype=np.float32), (
                np.array(batch_anchor_locations, dtype=np.float32),
                np.array(batch_anchor_labels, dtype=np.float32),
            )



    # while 1:
    #     random_indexes = np.random.randint(
    #         low=0, high=len(imgs) - 1, size=cfg.N_DATA_EPOCHS
    #     )
    #     batch_imgs = np.zeros(
    #         (len(random_indexes), img_size[0], img_size[1], 1), dtype=np.float32
    #     )
    #     batch_anchor_labels = np.zeros(
    #         (
    #             len(random_indexes),
    #             img_size[0] // N_SUB,
    #             img_size[1] // N_SUB,
    #             N_ANCHORS * N_RATIOS,
    #         ),
    #         dtype=np.float32,
    #     )
    #     batch_anchor_locations = np.zeros(
    #         (
    #             len(random_indexes),
    #             img_size[0] // N_SUB,
    #             img_size[1] // N_SUB,
    #             4 * N_ANCHORS * N_RATIOS,
    #         ),
    #         dtype=np.float32,
    #     )
    #
    #     for k, random_index in enumerate(random_indexes):
    #         img = imgs[random_index]
    #         bbox_dataset = bbox_datasets[random_index]
    #
    #         ious = calculate_bbox_intesect_over_union(
    #             anchors, index_anchors_valid, bbox_dataset, img
    #         )
    #
    #         labels, anchor_argmax_ious = evaluate_ious(
    #             anchors,
    #             index_anchors_valid,
    #             ious,
    #             bbox_dataset,
    #             img,
    #             cfg.POS_IOU_THRESHOLD,
    #             cfg.NEG_IOU_THRESHOLD,
    #         )
    #
    #         anchor_labels = create_samples_for_training(
    #             anchors,
    #             labels,
    #             bbox_dataset,
    #             img,
    #             debug=False,
    #         )
    #
    #         anchor_labels = np.reshape(
    #             anchor_labels,
    #             (img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS),
    #         )
    #
    #         anchor_locations = parametrize_anchor_box_properties(
    #             anchors, anchor_argmax_ious, labels, ious, bbox_dataset, img
    #         )
    #
    #         anchor_locations = np.reshape(
    #             anchor_locations,
    #             (img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS),
    #         )
    #
    #         anchor_labels = anchor_labels.astype(np.float32)
    #
    #         batch_imgs[k, :, :, 0] = img
    #
    #         batch_anchor_labels[k, :, :, :] = anchor_labels
    #         batch_anchor_locations[k, :, :, :] = anchor_locations
    #
    #     yield batch_imgs, (batch_anchor_locations, batch_anchor_labels)
