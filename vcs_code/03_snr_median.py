import numpy as np


def snr(img, smask, nmask=None):
    fg_mean = np.median(img[smask > 0])

    if nmask is None:
        nmask = smask

    bg_std = img[nmask > 0].std()
    return float(fg_mean / bg_std)
