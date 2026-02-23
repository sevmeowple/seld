import numpy as np


def process_foa_input_sed_doa_labels(feat, label):
    mel_bins = 64
    # nb_ch = 1
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label


def process_foa_input_sed_doa(feat):
    mel_bins = 64
    # nb_ch = 1
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat


def process_foa_input_128d_sed_doa_labels(feat, label):
    mel_bins = 128
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label


def process_foa_input_ssast_data_labels(feat, label):
    mel_bins = 128
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    feat = feat[0, :, :]
    return feat, label


def process_foa_input_sed_labels(feat, label):
    nb_classes = 13
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label[:, :nb_classes]
