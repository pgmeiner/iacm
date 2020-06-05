from math import log2
from typing import List
import numpy as np
import pandas as pd


def entropy(p: List[float]) -> float:
    ent = 0
    for v in p:
        if v > 0:
            ent = ent - v * log2(v)

    return ent


def multi_information(x: List[float], y: List[float], y0: List[float], y1: List[float], joint_distr: List[float]) \
        -> float:
    return entropy(x) + entropy(y) + entropy(y0) + entropy(y1) - entropy(joint_distr)


def kl_divergence(p: List[float], q: List[float]) -> float:
    res = 0.0
    for p_v, q_v in zip(p, q):
        if q_v == 0:
            return np.inf
        elif p_v > 0:
            res = res + p_v * log2(p_v / q_v)
    return res


def get_empirical_distribution(data: List, nr_bins=-1) -> List[float]:
    alphabet = np.unique(data)
    if nr_bins == -1:
        nr_bins = len(alphabet)
    p_, _ = np.histogram(data, bins=nr_bins, density=False)
    p = p_ / np.sum(p_)
    return p


def kl_divergence_x_y(data: pd.DataFrame, nr_bins_x: int, nr_bins_y: int) -> float:
    p_x = get_empirical_distribution(data['X'], nr_bins_x)
    p_y = get_empirical_distribution(data['Y'], nr_bins_y)
    return kl_divergence(p_x, p_y)
