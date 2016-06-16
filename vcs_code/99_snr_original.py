#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=no-member
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-03 11:29:22


def snr(img, smask, nmask=None, erode=True, fglabel=1):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:
    .. math::
        \text{SNR} = \frac{\mu_F}{\sigma_F},
    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.
    Alternatively, a background mask containing only noise can be provided.
    This must be an air mask around the head, and it should not contain artifacts.
    The computation is done following the eq. A.12 of [Dietrich2007]_, which
    includes a correction factor in the estimation of the standard deviation of
    air and its Rayleigh distribution:
    .. math::
        \text{SNR} = \frac{\mu_F}{\sqrt{\frac{2}{4-\pi}}\,\sigma_\text{air}}.
    :param numpy.ndarray img: input data
    :param numpy.ndarray fgmask: input foreground mask or segmentation
    :param numpy.ndarray bgmask: input background mask or segmentation
    :param bool erode: erode masks before computations.
    :param str fglabel: foreground label in the segmentation data.
    :param str bglabel: background label in the segmentation data.
    :return: the computed SNR for the foreground segmentation
    """
    fgmask = _prepare_mask(smask, fglabel, erode)
    bgmask = _prepare_mask(nmask, 1, erode) if nmask is not None else None

    fg_mean = np.median(img[fgmask > 0])
    if bgmask is None:
        bgmask = fgmask
        bg_mean = fg_mean
        # Manually compute sigma, using Bessel's correction (the - 1 in the normalizer)
        bg_std = np.sqrt(np.sum((img[bgmask > 0] - bg_mean) ** 2) / (np.sum(bgmask) - 1))
    else:
        bg_std = np.sqrt(2.0/(4.0 - pi)) * img[bgmask > 0].std(ddof=1)

    return float(fg_mean / bg_std)
