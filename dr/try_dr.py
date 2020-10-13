import numpy as np
from scipy.stats import binom, geom, hypergeom, nbinom, poisson
from typing import Tuple, Any
import pandas as pd
import os
# res tells you the results:
#    0: both directions
#    1: only correct direction
#    2: no direction
#   -1: only wrong direction inferred.
# example_rev tells you more about the instances of 0 and
# example_rev2 tells you more about the instances of -1.

# %simulates data from a discrete additive noise model.
# %
# %-X and Y are vectors (each num_samplesx1) containing the samples.
# %
# %-num_samples: number of samples.
# %
# %-n_distr can be
# %    'bino' for binornd
# %    'geo' for geornd
# %    'hypergeo' for hygernd
# %    'multin' for mnrnd
# %    'negbin' for nbinrnd
# %    'poisson' for poissrnd
# %    'custom' for a user-specific noise distribution. Then
# %        pars_n.p_n: a vector of probabilities (should sum to one)
# %        pars_n.n_values: a vector of values of n
# %-otherwise pars_n contains the parameters for the distribution.
# %
# %
# %
# %
# % CASE 1:
# %-X_distr can be 'custom'. Then
# %-pars_X contains
# %     pars_X.p_X: a vector of probabilities (should sum to one)
# %     pars_X.X_values: a vector of values of X
# %-fct_kind should be 'vector',
# %-fct is a vector (kx1) containing the function values on pars_X.X_values
# %     (thus they should have the same length).
# %
# %
# %CASE 2:
# %-X_distr can be
# %    'bino' for binornd
# %    'geo' for geornd
# %    'hypergeo' for hygernd
# %    'multin' for mnrnd
# %    'negbin' for nbinrnd
# %    'poisson' for poissrnd. Then
# %-pars_X contains the parameters for the distribution.
# %-fct_kind should be 'fct',
# %-fct should be a function (e.g. @(x) round(0.5*x.^2))
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Example
# %%%%%%%%%
# % num_samples=200;
# % fct_kind='vector';
# % fct=[0;3;4];
# % X_distr='custom';
# % pars_X.p_X=[0.6 0.15 0.25];
# % pars_X.X_values=[-3;1;2];
# % n_distr='custom';
# % pars_n.p_n=[0.1 0.3 0.3 0.2 0.1];
# % pars_n.n_values=[-2;-1;0;1;2];
# %
# % [X Y]=add_noise(num_samples, fct, X_distr, pars_X, n_distr, pars_n, fct_kind);
from dr.discrete_regression import fit_both_dir_discrete


def add_noise(num_samples, fct,
              X_distr, p_X, X_values, X_N, X_p, X_M, X_K, X_R, X_P, X_lambda,
              n_distr, p_n, n_values, n_N, n_p, n_M, n_K, n_R, n_P, n_lambda, fct_kind) -> Tuple[Any, Any]:
    noise_cdf = []
    if n_distr == 'custom':
        tmp = 0
        for i in range(1,len(p_n)):
            tmp = tmp + p_n[i]
            noise_cdf = noise_cdf + [tmp]
        eps = np.random.rand(num_samples)
        for i in range(0, num_samples):
            eps[i] = sum(eps[i] > noise_cdf)
        eps = np.take(n_values, eps.astype(int))
    elif n_distr == 'bino':
        eps = binom.rvs(n_N, n_p, size=num_samples)
    elif n_distr == 'geo':
        eps = geom.rvs(n_p, size=num_samples)
    elif n_distr == 'hypergeo':
        eps = hypergeom.rvs(n_M, n_K, n_N, size=num_samples)
    elif n_distr == 'negbin':
        eps = nbinom.rvs(n_R, n_P, size=num_samples)
    elif n_distr == 'poisson':
        eps = poisson.rvs(n_lambda, size=num_samples)

    X_cdf = []
    if X_distr == 'custom':
        tmp = 0
        for i in range(1, len(p_X)):
            tmp = tmp + p_X[i]
            X_cdf = X_cdf + [tmp]
        X = np.random.rand(num_samples)
        for i in range(0, num_samples):
            X[i] = sum(X[i] > X_cdf)
        if fct_kind == 'fct':
            X = np.take(X_values, X.astype(int))
            Y = fct[X.astype(int)] + eps
        elif fct_kind == 'vector':
            Y = fct[X.astype(int)] + eps
            X = np.take(X_values, X.astype(int))
    elif X_distr == 'bino':
        X = binom.rvs(X_N, X_p, size=num_samples) + 1
        Y = fct[X] + eps
    elif X_distr == 'geo':
        X = geom.rvs(X_p, size=num_samples) + 1
        Y = fct[X] + eps
    elif X_distr == 'hypergeo':
        X = hypergeom.rvs(X_M, X_K, X_N, size=num_samples) + 1
        Y = fct[X] + eps
    elif X_distr == 'negbin':
        X = nbinom.rvs(X_R, X_P, size=num_samples) + 1
        Y = fct[X] + eps
    elif X_distr == 'poisson':
        X = poisson.rvs(X_lambda, size=num_samples)
        Y = fct[X] + eps
    return X, Y


def print_result(res):
    total = len(res.values())
    both = round(sum([e == 0 for e in res.values()]) / total * 100, 2)
    correct = round(sum([e == 1 for e in res.values()]) / total * 100, 2)
    wrong = round(sum([e == -1 for e in res.values()]) / total * 100, 2)
    no_direction = round(sum([e == 2 for e in res.values()]) / total * 100, 2)
    print(f'{correct} % correct, {wrong} % wrong, {both} % both, {no_direction} % no direction')


if __name__ == "__main__":
    counter = 0
    counter2 = 0
    level = 0.05
    num_samples = 2000
    mod = dict()
    example_rev = dict()
    example_rev2 = dict()
    speicherX = dict()
    speicherY = dict()
    res = dict()

    for i in range(0, 1000):
        print(i)
        # alpha = 2*(np.random.rand(7,1) - 0.5*np.ones(7))
        # fct_kind='fct';
        # fct=@(x) round(alpha(1)+alpha(2)*x+alpha(3)*x.^2+alpha(4)*x.^3+alpha(5)*x.^4+alpha(6)*x.^5+alpha(7)*x.^6);
        fct_kind = 'vector'
        fct = np.round(np.random.randint(1, 15 + 1, 2000) - 8*np.ones(2000))
        # fct = round(randi(15,2000,1)-8*ones(2000,1))
        mod[1] = np.random.randint(1, 7 + 1)
        mod[2] = 1

        x_rand = np.array([0])
        x_rand = np.append(x_rand, np.sort(np.random.rand(6 - 1, 1)))
        x_rand = np.append(x_rand, np.array([1]))
        p_X = np.diff(x_rand)
        X_values = np.arange(1, 6 + 1)
        X_distr = 'custom'
        X_N = np.random.randint(1, 40 + 1)
        X_p = 0.8 * np.random.rand(1)[0] + 0.1
        X_M = np.random.randint(1, 40 + 1)
        X_K = np.random.randint(1, X_M + 1)
        X_R = np.random.randint(1, 20 + 1)
        X_P = 0.8 * np.random.rand(1)[0] + 0.1
        X_lambda = 10 * np.random.rand(1)[0]
        if mod[1] == 1:
            x_rand = np.array([0])
            x_rand = np.append(x_rand, np.sort(np.random.rand(6-1, 1)))
            x_rand = np.append(x_rand, np.array([1]))
            # x_rand = [0;sort(rand(6 - 1, 1));1]
            p_X = np.diff(x_rand)
            X_values = np.arange(1, 6+1)
            X_distr = 'custom'
        elif mod[1] == 2:
            x_rand = np.array([0])
            x_rand = np.append(x_rand, np.sort(np.random.rand(4 - 1, 1)))
            x_rand = np.append(x_rand, np.array([1]))
            p_X = np.diff(x_rand)
            X_values = np.arange(1, 4+1)
            X_distr = 'custom'
        elif mod[1] == 3:
            X_N = np.random.randint(1, 40 + 1)
            X_p = 0.8*np.random.rand(1)[0] + 0.1
            X_distr = 'bino'
        elif mod[1] == 4:
            X_p = 0.8*np.random.rand(1)[0] + 0.1
            X_distr = 'geo'
        elif mod[1] == 5:
            X_M = np.random.randint(1, 40 + 1)
            X_K = np.random.randint(1, X_M + 1)
            X_N = np.random.randint(1, X_K + 1)
            X_distr = 'hypergeo'
        elif mod[1] == 6:
            X_R = np.random.randint(1, 20 + 1)
            X_P = 0.8* np.random.rand(1)[0] + 0.1
            X_distr = 'negbin'
        elif mod[1] == 7:
            X_lambda = 10*np.random.rand(1)[0]
            X_distr = 'poisson'

        tmp = np.random.randint(1, 5 + 1)
        length = 2 * tmp + 1
        n_rand = np.array([0])
        n_rand = np.append(n_rand, np.sort(np.random.rand(length - 1, 1)))
        n_rand = np.append(n_rand, np.array([1]))
        p_n = np.diff(n_rand)
        n_values = np.arange(-tmp, tmp + 1)
        n_distr = 'custom'
        n_N = np.random.randint(1, 100 + 1)
        n_p = np.random.rand(1)[0]
        n_M = np.random.randint(1, 200 + 1)
        n_K = np.random.randint(1, n_M + 1)
        n_R = np.random.randint(1, 20 + 1)
        n_P = 0.8 * np.random.rand(1) + 0.1
        n_lambda = 10 * np.random.rand(1)[0]
        if mod[2] == 1:
            tmp = np.random.randint(1, 5 + 1)
            length = 2*tmp + 1
            n_rand = np.array([0])
            n_rand = np.append(n_rand, np.sort(np.random.rand(length - 1, 1)))
            n_rand = np.append(n_rand, np.array([1]))
            p_n = np.diff(n_rand)
            n_values = np.arange(-tmp, tmp+1)
            n_distr = 'custom'
        elif mod[2] == 2:
            n_rand = np.array([0])
            n_rand = np.append(n_rand, np.sort(np.random.rand(5 - 1, 1)))
            n_rand = np.append(n_rand, np.array([1]))
            p_n = np.diff(n_rand)
            n_values = np.arange(-2, 2+1)
            n_distr = 'custom'
        elif mod[2] == 3:
            n_N = np.random.randint(1, 100 + 1)
            n_p = np.random.rand(1)[0]
            n_distr = 'bino'
        elif mod[2] == 4:
            n_p = 0.8*np.random.rand(1)[0] + 0.1
            n_distr = 'geo'
        elif mod[2] == 5:
            n_M = np.random.randint(1, 200 + 1)
            n_K = np.random.randint(1, n_M + 1)
            n_N = np.random.randint(1, n_M + 1)
            n_distr = 'hypergeo'
        elif mod[2] == 6:
            n_R = np.random.randint(1, 20 + 1)
            n_P = 0.8*np.random.rand(1) + 0.1
            n_distr = 'negbin'
        elif mod[2] == 7:
            n_lambda = 10*np.random.rand(1)[0]
            n_distr = 'poisson'

        X, Y = add_noise(num_samples, fct,
                         X_distr, p_X, X_values, X_N, X_p, X_M, X_K, X_R, X_P, X_lambda,
                         n_distr, p_n, n_values, n_N, n_p, n_M, n_K, n_R, n_P, n_lambda, fct_kind)

        data = pd.DataFrame({'X': X, 'Y': Y})
        if not os.path.exists(f'../simulations/add_noise_a1'):
            os.makedirs(f'../simulations/add_noise_a1')
        filename = f'pair{i}.csv'
        data.to_csv(f'../simulations/add_noise_a1/{filename}', sep=" ", header=False, index=False)
        speicherX[i] = X
        speicherY[i] = Y

        fct_fw, p, fct_bw, p_bw = fit_both_dir_discrete(X, False, Y, False, level)
        if (p > level) and (p_bw > level):
            res[i] = 0
            counter = counter + 1
            example_rev['number{counter}'] = i
            example_rev['mod{counter}'] = mod[1]
            example_rev['X{counter}'] = np.unique(X)
            example_rev['fct{counter}'] = fct[np.unique(X)]
            example_rev['n_distr{counter}'] = p_n
            uni_X = np.unique(X)
            x_distr = dict()
            for j in range(1, len(uni_X)):
                x_distr[j] = sum(X == uni_X[j]) / num_samples
            example_rev['x_distr{counter}'] = x_distr
        elif (p > level) and (p_bw < level):
            res[i] = 1
        elif (p < level) and (p_bw > level):
            res[i] = -1
            counter2 = counter2 + 1
            example_rev2['number{counter2}'] = i
            example_rev2['mod{counter2}'] = mod[1]
            example_rev2['X{counter2}'] = np.unique(X)
            example_rev2['fct{counter2}'] = fct[np.unique(X)]
            example_rev2['n_distr{counter2}'] = p_n
            uni_X = np.unique(X)
            x_distr = dict()
            for j in range(1, len(uni_X)):
                x_distr[j] = sum( X == uni_X[j]) / num_samples
                example_rev2['x_distr{counter2}'] = x_distr
        elif (p < level) and (p_bw < level):
            res[i] = 2

        print_result(res)

    pass
