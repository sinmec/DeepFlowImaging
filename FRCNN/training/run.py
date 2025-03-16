import numpy as np
from training.create_anchors import create_anchors
from training.calculate_bbox_intesect_over_union import calculate_bbox_intesect_over_union
from training.evaluate_ious import evaluate_ious
from training.create_samples_for_training import create_samples_for_training
from training.parametrize_anchor_box_properties import parametrize_anchor_box_properties
from tqdm import tqdm


def generate_validation_data(imgs, bbox_datasets, img_size, N_SUB, N_ANCHORS, ANCHOR_SIZES, ANCHOR_RATIOS, N_RATIOS,
                             POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD, DEBUG, POS_RATIO, N_SAMPLES, SHOW_N_POS):

    # Creating the anchors
    anchors, index_anchors_valid = create_anchors(img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES, imgs[0])

    # Number of validation images
    N_validation = imgs.shape[0]

    # Initializing the arrays
    batch_imgs             = np.zeros((N_validation, img_size[0], img_size[1], 1), dtype=np.float64)
    batch_anchor_labels    = np.zeros((N_validation, img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS),     dtype=np.float64)
    batch_anchor_locations = np.zeros((N_validation, img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS), dtype=np.float64)

    index = 0
    for img, bbox_dataset in tqdm(zip(imgs, bbox_datasets), desc="Generating validation data"):

        # Calculating anchor/bbox_dataset IoUs
        ious = calculate_bbox_intesect_over_union(anchors, index_anchors_valid, bbox_dataset, img)

        # Evaluating if the anchors are valid or invalid based on the IoUs
        labels, anchor_argmax_ious = evaluate_ious(anchors, index_anchors_valid, ious, bbox_dataset, img, POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD, debug=DEBUG)

        # Creating the samples for training
        anchor_labels = create_samples_for_training(anchors, index_anchors_valid, anchor_argmax_ious, labels, ious, bbox_dataset, img,
                                                    POS_RATIO, N_SAMPLES, SHOW_N_POS, debug=DEBUG)

        # Reshaping the anchor labels to follow the image/sub-image coordinates
        anchor_labels = np.reshape(anchor_labels, (img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS))

        # Parametrizing the anchor box properties
        anchor_locations = parametrize_anchor_box_properties(anchors, anchor_argmax_ious, labels, ious, bbox_dataset, img)

        # Reshaping the anchor locations to follow the image/sub-image coordinates
        anchor_locations = np.reshape(anchor_locations, (img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS))

        # Converting to float
        anchor_labels = anchor_labels.astype(np.float64)

        # Storing images
        batch_imgs[index, :, :, 0] = img

        # Updating anchor labels and properties(locations)
        batch_anchor_labels[index,:,:,:] = anchor_labels
        batch_anchor_locations[index,:,:,:] = anchor_locations

        index += 1

    # Returning samples for model validation
    return batch_imgs, [batch_anchor_labels, batch_anchor_locations]


def input_generator(imgs, bbox_datasets, img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES, N_DATA_EPOCHS, N_ANCHORS, N_RATIOS,
                    POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD, DEBUG, POS_RATIO, N_SAMPLES, SHOW_N_POS):

    # Creating the anchors
    anchors, index_anchors_valid = create_anchors(img_size, N_SUB, ANCHOR_RATIOS, ANCHOR_SIZES, imgs[0])


    while True:
        # Picking a random number of images for training
        random_indexes = np.random.randint(low=0, high=len(imgs)-1, size=N_DATA_EPOCHS)
        # Initializing the arrays
        batch_imgs             = np.zeros((len(random_indexes), img_size[0], img_size[1], 1), dtype=np.float64)
        batch_anchor_labels    = np.zeros((len(random_indexes), img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS),     dtype=np.float64)
        batch_anchor_locations = np.zeros((len(random_indexes), img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS), dtype=np.float64)

        # Looping over the selected indexes and generating the input dataset
        for k, random_index in enumerate(random_indexes):

            # Retriegving the image and the bbox-values
            img = imgs[random_index]
            bbox_dataset = bbox_datasets[random_index]

            # Calculating anchor/bbox_dataset IoUs
            ious = calculate_bbox_intesect_over_union(anchors, index_anchors_valid, bbox_dataset, img)

            # Evaluating if the anchors are valid or invalid based on the IoUs
            labels, anchor_argmax_ious = evaluate_ious(anchors, index_anchors_valid, ious, bbox_dataset, img, POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD, debug=DEBUG)

            # Creating the samples for training

            anchor_labels = create_samples_for_training(anchors, index_anchors_valid, anchor_argmax_ious, labels, ious, bbox_dataset, img,
                                                        POS_RATIO, N_SAMPLES, SHOW_N_POS, debug=False)

            # Reshaping the anchor labels to follow the image/sub-image coordinates
            anchor_labels = np.reshape(anchor_labels, (img_size[0] // N_SUB, img_size[1] // N_SUB, N_ANCHORS * N_RATIOS))

            # Parametrizing the anchor box properties
            anchor_locations = parametrize_anchor_box_properties(anchors, anchor_argmax_ious, labels, ious, bbox_dataset, img)

            # Reshaping the anchor locations to follow the image/sub-image coordinates
            anchor_locations = np.reshape(anchor_locations, (img_size[0] // N_SUB, img_size[1] // N_SUB, 4 * N_ANCHORS * N_RATIOS))

            # Converting to float
            anchor_labels = anchor_labels.astype(np.float64)

            # Storing images
            batch_imgs[k, :, :, 0] = img

            print(random_indexes)

            # Updating anchor labels and properties(locations)
            batch_anchor_labels[k,:,:,:] = anchor_labels
            batch_anchor_locations[k,:,:,:] = anchor_locations

            # Returning/Yielding samples for model training
            yield batch_imgs, (batch_anchor_labels, batch_anchor_locations)
