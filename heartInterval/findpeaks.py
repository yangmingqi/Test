# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:20:57 2017

@author: laoyang
"""
import numpy as np


def findpeaks(x, mph=None, mpd=1, threshold=0,
              kpsh=False, sort_by_height=False):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    """
    x = np.atleast_1d(x).astype('float64')
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    ind = np.where((np.hstack((dx, 0)) <= 0) &
                   (np.hstack((0, dx)) > 0))[0]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(
            np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    ind = ind[np.argsort(x[ind])][::-1]
    if ind.size and mpd > 1:
        # ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = ind[~idel]
        if not sort_by_height:
            ind = np.sort(ind)
    return x[ind], ind