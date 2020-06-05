from math import log2
from typing import List
import numpy as np


def entropy(p):
    ent = 0
    for v in p:
        if v > 0:
            ent = ent - v * log2(v)

    return ent


def multi_information(x, y, y0, y1, joint_distr):
    return entropy(x) + entropy(y) + entropy(y0) + entropy(y1) - entropy(joint_distr)


def KL_Dist(P: List[float], Q: List[float]) -> float:
    res = 0.0
    for p, q in zip(P, Q):
        if q == 0:
            return np.inf
        elif p > 0:
            res = res + p * log2(p / q)
    return res


def get_empirical_distribution(data: List, nr_bins=-1) -> List[float]:
    alphabet = np.unique(data)
    if nr_bins == -1:
        nr_bins = len(alphabet)
    p_, _ = np.histogram(data, bins=nr_bins, density=False)
    p = p_ / np.sum(p_)
    return p


def KL_Dist_X_Y(data, nr_bins_x: int, nr_bins_y: int) -> float:
    p_X = get_empirical_distribution(data['X'], nr_bins_x)
    p_Y = get_empirical_distribution(data['Y'], nr_bins_y)
    return KL_Dist(p_X, p_Y)
