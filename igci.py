#####
##
##     Python implementation of causal inference algorithm originally written in Matlab
##     From paper:
##          "Inferring deterministic causal relations." (UAI-2010)
##           P. Daniushis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel, K. Zhang, B. Scholkopf
##           http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf
##
##     Author: S. Bachish
##
#####

import numpy as np
from scipy.special import psi


def igci(x, y, refMeasure=2, estimator=1):
    ###
    ## Performs causal inference in a deterministic scenario
    #
    ##   refMeasure - reference measure to use:
    #           1: uniform,
    #           2: Gaussian
    #
    ##   estimator -  estimator to use:
    #           1: entropy (eq. (12) in [1]),
    #           2: integral approximation (eq. (13) in [1])
    #
    ## OUTPUT:
    ##      f < 0:       the method prefers the causal direction x -> y
    ##      f > 0:       the method prefers the causal direction y -> x
    ###

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # check input arguments
    if len(x.shape) != 1 and min(x.shape[0], x.shape[1]) != 1:
        raise Exception('Dimensionality of x must be 1')
    x = x.ravel()
    if x.shape[0] < 20:
        raise Exception('Not enough observations in x (must be > 20)')
    m = x.shape[0]

    if len(y.shape) != 1 and min(y.shape[0], y.shape[1]) != 1:
        raise Exception('Dimensionality of y must be 1')
    y = y.ravel()
    if y.shape[0] < 20:
        raise Exception('Not enough observations in y (must be > 20)')

    if refMeasure == 1:
        # uniform reference measure
        x = (x - min(x)) / (max(x) - min(x))
        y = (y - min(y)) / (max(y) - min(y))
    elif refMeasure == 2:
        # Gaussian reference measure
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    else:
        print('Warning: unknown reference measure - no scaling applied')

    if estimator == 1:
        # difference of entropies
        x1 = np.sort(x)
        indXs = np.argsort(x)  # [x1,indXs] = sort(x);
        y1 = np.sort(y)
        indYs = np.argsort(y)

        n1 = len(x1)
        hx = 0.0

        for i in range(0, n1 - 1):
            delta = x1[i + 1] - x1[i]
            if delta:    hx = hx + np.log(abs(delta))

        hx = hx / (n1 - 1) + psi(n1) - psi(1)

        n2 = len(y1)
        hy = 0.0
        for i in range(0, n2 - 1):
            delta = y1[i + 1] - y1[i]
            if delta:    hy = hy + np.log(abs(delta))

        hy = hy / (n2 - 1) + psi(n2) - psi(1)
        f = hy - hx

    elif estimator == 2:
        # integral-approximation based estimator
        a, b = 0, 0

        ind1 = np.argsort(x)
        ind2 = np.argsort(y)

        for i in range(0, m - 1):  # for i=1:m-1:
            X1 = x[ind1[i]]
            X2 = x[ind1[i + 1]]
            Y1 = y[ind1[i]]
            Y2 = y[ind1[i + 1]]
            if (X2 != X1) and (Y2 != Y1):    a += np.log(abs((Y2 - Y1) / (X2 - X1)))

            X1 = x[ind2[i]]
            X2 = x[ind2[i + 1]]
            Y1 = y[ind2[i]]
            Y2 = y[ind2[i + 1]]
            if (Y2 != Y1) and (X2 != X1):    b += np.log(abs((X2 - X1) / (Y2 - Y1)))

        f = (a - b) / m

    else:
        raise Exception('Unknown estimator')

    return f