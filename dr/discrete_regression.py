# Python  implementation of Causal Inference on Discrete Data using Additive Noise Models.
import random
from typing import List, Tuple
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import numpy as np


def dr(X: List, Y: List, alpha: float):
    fct_fw, p_fw, fct_bw, p_bw = fit_both_dir_discrete(X, False, Y, False, alpha)
    return p_fw, p_bw


def fit_both_dir_discrete(X: List, cyclic_X: bool, Y: List, cyclic_Y: bool, alpha: float):
    # -fits a discrete additive noise model in both directions X->Y and Y->X.
    # -X and Y should both be of size (n,1),
    # -cycX is 1 if X should be modelled as a cyclic variable and 0 if not
    # -cycY is 1 if Y should be modelled as a cyclic variable and 0 if not
    # -alpha denotes the significance level of the independent test after which
    # the algorithm should stop looking for a solution
    # -doplots=1 shows a plot of the function and the residuals for each
    # iteration (at the end there will be plots in each case)
    # -example:
    # pars.p_X=[0.1 0.3 0.1 0.1 0.2 0.1 0.1];pars.X_values=[-3;-2;-1;0;1;3;4];
    # pars2.p_n=[0.2 0.5 0.3];pars2.n_values=[-1;0;1];
    # [X Y]=add_noise(500,@(x) round(0.5*x.^2),'custom',pars,'custom',pars2, 'fct');
    # [fct1 p_val1 fct2 p_val2]=fit_both_dir_discrete(X,0,Y,0,0.05,0);
    #
    # -please cite
    # Jonas Peters, Dominik Janzing, Bernhard Schoelkopf (2010): Identifying Cause and Effect on Discrete Data using Additive Noise Models,
    # in Y.W. Teh and M. Titterington (Eds.), Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS) 2010,
    # JMLR: W&CP 9, pp 597-604, Chia Laguna, Sardinia, Italy, May 13-15, 2010,
    # -if you have problems, send me an email:
    # jonas.peters ---at--- tuebingen.mpg.de
    #
    # Copyright (C) 2010 Jonas Peters
    #
    #    This file is part of discrete_anm.

    #    discrete_anm is free software: you can redistribute it and/or modify
    #    it under the terms of the GNU General Public License as published by
    #    the Free Software Foundation, either version 3 of the License, or
    #    (at your option) any later version.
    #
    #    discrete_anm is distributed in the hope that it will be useful,
    #    but WITHOUT ANY WARRANTY; without even the implied warranty of
    #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #    GNU General Public License for more details.
    #
    #    You should have received a copy of the GNU General Public License
    #    along with discrete_anm.  If not, see <http://www.gnu.org/licenses/>.

    if not cyclic_Y:
        fct_fw, p_val_fw = fit_discrete(X, Y, alpha)
    else:
        fct_fw, p_val_fw = fit_discrete_cyclic(X, Y, alpha)

    if not cyclic_X:
        fct_bw, p_val_bw = fit_discrete(Y, X, alpha)
    else:
        fct_bw, p_val_bw = fit_discrete_cyclic(Y, X, alpha)

    return fct_fw, p_val_fw, fct_bw, p_val_bw


def fit_discrete(X: List, Y: List, alpha: float):
    # -please cite
    # Jonas Peters, Dominik Janzing, Bernhard Schoelkopf (2010): Identifying Cause and Effect on Discrete Data using Additive Noise Models,
    # in Y.W. Teh and M. Titterington (Eds.), Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS) 2010,
    # JMLR: W&CP 9, pp 597-604, Chia Laguna, Sardinia, Italy, May 13-15, 2010,
    #
    # -if you have problems, send me an email:
    # jonas.peters ---at--- tuebingen.mpg.de
    #
    # Copyright (C) 2010 Jonas Peters

    num_iter = 10
    num_pos_fct = min(max(Y) - min(Y), 20)

    # rescaling:
    # X_new takes values from 1...X_new_max
    # Y_values are everything between Y_min and Y_max
    X_values, aa, X_new = np.unique(X, return_index=True, return_inverse=True)
    Y_values = np.arange(min(Y), max(Y) + 1)
    cand = dict()
    pos_fct = dict()
    p_val_comp = dict()
    p_val_comp2 = dict()

    if len(X_values) == 1 or len(Y_values) == 1:
        fct = np.ones(len(X_values)) * Y_values[0]
        p_val = 1
    else:
        p, _ = np.histogramdd((X, Y), bins=(len(X_values), len(Y_values)))

        fct = np.zeros(len(X_values))
        for i in range(0, len(X_values)):
            # a = np.sort(p[i, :], axis=0)
            b = np.argsort(p[i, :], axis=0)
            for k in range(0, p.shape[1]):
                if k != b[len(b)-1]:
                    p[i, k] = p[i, k] + 1 / (2 * abs(k - b[len(b)-1]))
                else:
                    p[i, k] = p[i, k] + 1
            # a = np.sort(p[i, :], axis=0)
            b = np.argsort(p[i, :], axis=0)
            cand[i] = b
            fct[i] = Y_values[b[len(b)-1]]

        yhat = fct[X_new]
        eps = Y - yhat
        if len(np.unique(eps)) == 1:
            print('Warning!! there is a deterministic relation between X and Y')
            p_val = 1
        else:
            p_val, _ = chi_sq_quant(eps, X, len(np.unique(eps)), len(X_values))
        i = 0
        while (p_val < alpha) & (i < num_iter):
            for j_new in np.random.permutation(len(X_values)):
                for j in range(0, int(num_pos_fct)):
                    pos_fct[j] = fct
                    pos_fct[j][j_new] = Y_values[cand[j_new][len(cand[j_new]) - 1 - j]]
                    yhat = pos_fct[j][X_new]
                    eps = Y - yhat
                    p_val_comp[j], p_val_comp2[j] = chi_sq_quant(eps, X, len(np.unique(eps)), len(X_values))

                aa = np.max([v for v in p_val_comp.values()])
                j_max = np.argmax([v for v in p_val_comp.values()])
                if aa < 1e-3:
                    # aa = np.min([v for v in p_val_comp2.values()])
                    j_max = np.argmin([v for v in p_val_comp2.values()])

                fct = pos_fct[j_max]
                yhat = fct[X_new]
                eps = Y - yhat
                p_val, _ = chi_sq_quant(eps, X, len(np.unique(eps)), len(X_values))
            i = i + 1
        fct = fct + np.round(np.mean(eps))
    return fct, p_val


def fit_discrete_cyclic(X: List, Y: List, alpha: float):
    # -please cite
    # Jonas Peters, Dominik Janzing, Bernhard Schoelkopf (2010): Identifying Cause and Effect on Discrete Data using Additive Noise Models,
    # in Y.W. Teh and M. Titterington (Eds.), Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS) 2010,
    # JMLR: W&CP 9, pp 597-604, Chia Laguna, Sardinia, Italy, May 13-15, 2010,
    #
    # -if you have problems, send me an email:
    # jonas.peters ---at--- tuebingen.mpg.de
    #
    # Copyright (C) 2010 Jonas Peters

    num_iter = 10
    num_pos_fct = int(np.ceil(min(max(Y) - min(Y), 10)))

    # rescaling:
    # X_new takes values from 1...X_new_max
    # Y_values are everything between Y_min and Y_max
    X_values, aa, X_new = np.unique(X, return_index=True, return_inverse=True)
    Y_values = np.arange(min(Y), max(Y) + 1)
    cand = dict()
    pos_fct = dict()
    p_val_comp = dict()
    p_val_comp2 = dict()

    if len(X_values) == 1 or len(Y_values) == 1:
        fct = np.ones(len(X_values)) * Y_values[0]
        p_val = 1
    else:
        p, _ = np.histogramdd((X, Y), bins=(len(X_values), len(Y_values)))

        fct = np.zeros(len(X_values))
        for i in range(0, len(X_values)):
            # a = np.sort(p[i, :], axis=0)
            b = np.argsort(p[i, :], axis=0)
            for k in range(0, p.shape[1]):
                if k != b[len(b)-1]:
                    p[i, k] = p[i, k] + 1 / (2 * abs(k - b[len(b)-1]))
                else:
                    p[i, k] = p[i, k] + 1
            # a = np.sort(p[i, :], axis=0)
            b = np.argsort(p[i, :], axis=0)
            cand[i] = b
            fct[i] = Y_values[b[len(b) - 1]]

        yhat = fct[X_new]
        eps = (Y - yhat) % (max(Y) - min(Y) + 1)
        p_val, _ = chi_sq_quant(eps, X, len(np.unique(eps)), len(X_values))
        i = 0
        while (p_val < alpha) & (i < num_iter):
            for j_new in np.random.permutation(len(X_values)):
                for j in range(0, num_pos_fct):
                    pos_fct[j] = fct
                    pos_fct[j][j_new] = Y_values[cand[j_new][len(cand[j_new]) - 1 - j]]
                    yhat = pos_fct[j][X_new]
                    eps = (Y - yhat) % (max(Y) - min(Y) + 1)
                    p_val_comp[j], p_val_comp2[j] = chi_sq_quant(eps, X, len(np.unique(eps)), len(X_values))

                aa = np.max([v for v in p_val_comp.values()])
                j_max = np.argmax([v for v in p_val_comp.values()])
                if aa < 1e-3:
                    # aa = np.min([v for v in p_val_comp2.values()])
                    j_max = np.argmin([v for v in p_val_comp2.values()])

                fct = pos_fct[j_max]
                yhat = fct[X_new]
                eps = (Y - yhat) % (max(Y) - min(Y) + 1)
                p_val, _ = chi_sq_quant(eps, X, len(np.unique(eps)), len(X_values))
            i = i + 1
    return fct, p_val


def chi_sq_quant(x, y, num_states_x: int, num_states_y: int) -> Tuple[float, float]:
    a, b, x = np.unique(x, return_inverse=True, return_index=True)
    a, b, y = np.unique(y, return_inverse=True, return_index=True)
    x = x - min(x)
    y = y - min(y)
    n_star = np.zeros((num_states_x, num_states_y))
    tmp = np.zeros((num_states_x, num_states_y))

    if num_states_x == 1 or num_states_y == 1:
        p_val = 1
        T = 0
    else:
        n_mat, _ = np.histogramdd((x, y), bins=(num_states_x, num_states_y))
        p = np.sum(n_mat, axis=1)
        w = np.sum(n_mat, axis=0)
        nullerp = sum(p == 0)
        nullerw = sum(w == 0)
        for i in range(0, num_states_x):
            for j in range(0, num_states_y):
                n_star[i, j] = p[i] * w[j] / len(x)
                if n_star[i, j] > 0:
                    tmp[i, j] = pow(n_mat[i, j] - n_star[i, j], 2) / n_star[i, j]
                else:
                    tmp[i, j] = 0
        T = sum(sum(tmp))
        p_val = 1 - chi2.cdf(T, (num_states_x - 1 - nullerp) * (num_states_y - 1 - nullerw))
    return p_val, T


def test_chi_sq_quant():
    X = [random.randint(1, 10) for i in range(1000)]
    Z = [random.randint(1, 10) for i in range(1000)]
    Y = [X[i] + random.randint(1, 3) for i in range(1000)]
    p, T = chi_sq_quant(x=X, y=Y, num_states_x=len(np.unique(X)), num_states_y=len(np.unique(Y)))
    if p < 0.05:
        print("X and Y dependent")
    p, T = chi_sq_quant(x=X, y=Z, num_states_x=len(np.unique(X)), num_states_y=len(np.unique(Z)))
    if p >= 0.05:
        print("X and Z independent")


if __name__ == "__main__":
    test_chi_sq_quant()
    X = [random.randint(1, 10) for i in range(1000)]
    Y = [X[i] + random.randint(1, 3) for i in range(1000)]
    print(dr(X, Y, 0.05))
    print(dr(Y, X, 0.05))
