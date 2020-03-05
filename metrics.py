from math import log2
from typing import List

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
            return 1000000.0
        elif p > 0:
            res = res + p * log2(p / q)
    return res
