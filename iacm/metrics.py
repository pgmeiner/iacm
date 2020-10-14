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
    p_x, p_y = get_embedded_empirical_distributions(data[data.columns[0]], data[data.columns[1]], nr_bins)
    return kl_divergence(p_x, p_y)


def get_kl_between_x_y(model_results: Dict[str, Any], base_x: int, base_y: int) -> float:
    if 'p_hat' in model_results:
        p_hat = model_results['p_hat']
        p_x = get_distr_x(p_hat, base_x)
        p_y = get_distr_y(p_hat, base_y)
        return kl_divergence(p_x, p_y)
    else:
        return np.inf


def get_distr_y(p_hat: Dict[str, float], base_y: int) -> List[float]:
    distr = []
    for y in range(0, base_y):
        y_sum = 0
        for k, v in p_hat.items():
            if str(y) == k[1]:
                y_sum = y_sum + v
        distr.append(y_sum)

    return [dist / sum(distr) for dist in distr]
    distr = [p_hat['0000'] + p_hat['0001'] + p_hat['0010'] + p_hat['0011'] + p_hat['1000'] + p_hat['1001'] +
             p_hat['1010'] + p_hat['1011'],
             p_hat['0100'] + p_hat['0101'] + p_hat['0110'] + p_hat['0111'] + p_hat['1100'] + p_hat['1101'] +
             p_hat['1110'] + p_hat['1111']]
    return distr / sum(distr)


def get_distr_x(p_hat: Dict[str, float], base_x: int) -> List[float]:
    distr = []
    for x in range(0, base_x):
        x_sum = 0
        for k, v in p_hat.items():
            if str(x) == k[0]:
                x_sum = x_sum + v
        distr.append(x_sum)

    return [dist / sum(distr) for dist in distr]

    distr = [p_hat['0000'] + p_hat['0001'] + p_hat['0010'] + p_hat['0011'] + p_hat['0100'] + p_hat['0101'] +
             p_hat['0110'] + p_hat['0111'],
             p_hat['1000'] + p_hat['1001'] + p_hat['1010'] + p_hat['1011'] + p_hat['1100'] + p_hat['1101'] +
             p_hat['1110'] + p_hat['1111']]
    return distr / sum(distr)


def get_distr_xy(p: Dict[str, float], base_x: int, base_y: int) -> List[float]:
    distr = []
    for x in range(0, base_x):
        for y in range(0, base_y):
            xy_sum = 0
            for k, v in p.items():
                xy = str(x) + str(y)
                if xy == k[:2]:
                   xy_sum = xy_sum + v
            distr.append(xy_sum)

    return [dist / sum(distr) for dist in distr]


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


def get_local_error(results: Dict[str, Any]) -> float:
    if 'kl_p_tilde_p_hat' in results:
        return results['kl_p_tilde_p_hat']
    else:
        return np.inf


def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1):
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r


def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc


def forward_auc(labels, predictions):
    target_one = [1 if x == 1 else 0 for x in labels]
    score = auc(target_one, predictions)
    return score


def reverse_auc(labels, predictions):
    target_neg_one = [1 if x == -1 else 0 for x in labels]
    neg_predictions = [-x for x in predictions]
    score = auc(target_neg_one, neg_predictions)
    return score


def bidirectional_auc(labels, predictions):
    score_forward = forward_auc(labels, predictions)
    score_reverse = reverse_auc(labels, predictions)
    score = (score_forward + score_reverse) / 2.0
    return score
