from math import pi
import numpy as np


def snr(img, smask, nmask=None):
    fg_mean = np.median(img[smask > 0])

    if nmask is None:
        nmask = smask
        # Manually compute sigma, using Bessel's correction (the - 1 in the normalizer)
        bg_mean = np.median(img[nmask > 0])
        bg_std = np.sqrt(np.sum((img[nmask > 0] - bg_mean) ** 2) / (np.sum(nmask) - 1))
    else:
        bg_std = np.sqrt(2.0/(4.0 - pi)) * img[nmask > 0].std(ddof=1)
    return float(fg_mean / bg_std)
