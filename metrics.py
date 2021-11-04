import numpy as np
from sklearn.metrics import mutual_info_score


def calc_snr(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def calc_system_distance(h, h_hat):

    # calculation of system distance in decibel

    # use n vector of the same length
    L = min(len(h), len(h_hat))
    h = h[1: L]
    h_hat = h_hat[1: L]

    # calculate coefficient error
    Delta_h = h - h_hat;

    # calculate power ratio
    D = sum(Delta_h** 2) / sum(h** 2);

    return 10 * np.log10(D);


def calc_erle(e, y, len, start):
    # Calculation of echo return loss enhancement in decibel

    Lx = len(e)
    ERLE = np.ones(Lx, 1)

    for k in range(start, Lx):

        ERLE[k] = np.mean(y[k - len + 1:k]** 2) / np.mean(e[k - len + 1: k]** 2)

    return 10 * np.log10(ERLE)


def calc_mutual_information(x, y, bins):

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi