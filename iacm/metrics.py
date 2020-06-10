from math import log2
from typing import List, Tuple, Dict, Any
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


def get_embedded_empirical_distributions(x: List, y: List, nr_bins=-1) -> Tuple[List[float], List[float]]:
    alphabet_x = np.unique(x)
    alphabet_y = np.unique(y)
    alphabet_xy = np.union1d(alphabet_x, alphabet_y)
    if nr_bins == -1:
        nr_bins = len(alphabet_xy)
    if len(alphabet_x) >= len(alphabet_y):
        p_x, bins = np.histogram(x, bins=nr_bins, range=(alphabet_xy.min(), alphabet_xy.max()), density=False)
        p_y, _ = np.histogram(y, bins=bins, range=(alphabet_xy.min(), alphabet_xy.max()), density=False)
    else:
        p_y, bins = np.histogram(y, bins=nr_bins, range=(alphabet_xy.min(), alphabet_xy.max()), density=False)
        p_x, _ = np.histogram(x, bins=bins, range=(alphabet_xy.min(), alphabet_xy.max()), density=False)
    return p_x / np.sum(p_x), p_y / np.sum(p_y)


def kl_divergence_x_y(data: pd.DataFrame, nr_bins: int) -> float:
    p_x, p_y = get_embedded_empirical_distributions(data['X'], data['Y'], nr_bins)
    return kl_divergence(p_x, p_y)


def get_kl_between_x_y(model_results: Dict[str, Any]) -> float:
    p_hat = model_results['p_hat']
    p_x = get_distr_x(p_hat)
    p_y = get_distr_y(p_hat)
    return kl_divergence(p_x, p_y)


def get_distr_y(p_hat: Dict[str, float]) -> List[float]:
    distr = [p_hat['0000'] + p_hat['0001'] + p_hat['0010'] + p_hat['0011'] + p_hat['1000'] + p_hat['1001'] +
             p_hat['1010'] + p_hat['1011'],
             p_hat['0100'] + p_hat['0101'] + p_hat['0110'] + p_hat['0111'] + p_hat['1100'] + p_hat['1101'] +
             p_hat['1110'] + p_hat['1111']]
    return distr / sum(distr)


def get_distr_x(p_hat: Dict[str, float]) -> List[float]:
    distr = [p_hat['0000'] + p_hat['0001'] + p_hat['0010'] + p_hat['0011'] + p_hat['0100'] + p_hat['0101'] +
             p_hat['0110'] + p_hat['0111'],
             p_hat['1000'] + p_hat['1001'] + p_hat['1010'] + p_hat['1011'] + p_hat['1100'] + p_hat['1101'] +
             p_hat['1110'] + p_hat['1111']]
    return distr / sum(distr)


def local_error(p_nom: float, p_denom: float, s: float) -> float:
    if p_denom > 0 and p_nom > 0:
        return 1 / s * p_nom * log2(p_nom / p_denom)
    elif p_nom == 0:
        return 0.0
    elif p_nom > 0:
        return np.inf


def calc_error(model_results: Dict[str, Any]) -> float:
    if "GlobalError" in model_results:
        return model_results["GlobalError"]
    else:
        return np.inf
