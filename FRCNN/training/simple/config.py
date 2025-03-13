# INPUT_FOLDER = "../../examples/example_dataset_FRCNN_PIV_subimage/Output"
INPUT_FOLDER = "/home/rafaelfc/Data/DeepFlowImaging/FRCNN/examples/example_dataset_FRCNN_synthetic_subimage_2025/Output/"

N_RATIO_LOSSES = 10.0
BATCH_SIZE_IMAGES = 50
N_EPOCHS = 10000
N_PATIENCE = 200
RANDOM_SEED = 13
N_SUB = 8

IMG_SIZE = (256, 256)  # Only square images are supported
MODE = "raw"

# ANCHOR_REAL_SIZE = [8, 16, 32, 48, 64, 96]
ANCHOR_REAL_SIZE = [16, 32, 48, 64, 96]
# ANCHOR_REAL_SIZE = [32, 64, 96]
ANCHOR_RATIOS = [0.8, 1.0, 1.2, 1.4, 1.6]
POS_IOU_THRESHOLD = 0.50
NEG_IOU_THRESHOLD = 0.1

N_FILTERS = 21
KERNEL_SIZE = (5, 5)
BATCH_SIZE = 100

SHOW_N_POS = False
POS_RATIO = 0.5
N_SAMPLES = 30

ADAM_LEARNING_RATE = 3.0e-4
# ADAM_LEARNING_RATE = 0.01
