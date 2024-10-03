import h5py as h5

import config as cfg


def save_model_configuration(filepath, N_SUB, ANCHOR_SIZES):
    FRCNN_model = h5.File(filepath, 'w')
    FRCNN_model.attrs['MODE'] = cfg.MODE
    FRCNN_model.attrs['POS_IOU_THRESHOLD'] = cfg.POS_IOU_THRESHOLD
    FRCNN_model.attrs['NEG_IOU_THRESHOLD'] = cfg.NEG_IOU_THRESHOLD
    FRCNN_model.attrs['IMG_SIZE'] = cfg.IMG_SIZE
    FRCNN_model.attrs['N_SUB'] = N_SUB
    FRCNN_model.attrs['ANCHOR_RATIOS'] = cfg.ANCHOR_RATIOS
    FRCNN_model.attrs['ANCHOR_SIZES'] = ANCHOR_SIZES
    FRCNN_model.close()
