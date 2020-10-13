import numpy as np
import pandas as pd
import os

# %res contains the results, the details of the reversible cases are stored
# %in f_counter, n_counter and p_counter
#
from dr.discrete_regression import fit_both_dir_discrete


def add_noise_cyclic(num_samples, fct, X_distr, noise_distr):
    # %simulates data from a discrete additive noise model.
    # %
    # %-X and Y are vectors (each num_samplesx1) containing the samples.
    # %
    # %-num_samples: number of samples.
    # %-X_distr: vector (1 x num_of_states_of_X) from which the input
    # %    distribution is sampled (should sum to 1)
    # %-fct: vector (num_of_states_of_X x 1) of function values (should have the
    # %    same length as X_distr)
    # %-noise_distr: vector (1 x num_of_states_of_Y) from which the additive
    # %    noise is sampled. This vector should the same number of components as
    # %    Y has states.
    num_states_X = len(X_distr)
    num_states_Y = len(noise_distr)
    X_values = np.arange(1, num_states_X+1) - 1
    noise_values = np.arange(0, (num_states_Y))
    noise_cdf = dict()
    X_cdf = dict()
    tmp = 0
    for i in range(0, len(noise_distr)):
        tmp = tmp + noise_distr[i]
        noise_cdf[i] = tmp
    tmp = 0
    for i in range(0, len(X_distr)):
        tmp = tmp + X_distr[i]
        X_cdf[i] = tmp
    X = np.random.rand(num_samples)
    for i in range(0, num_samples):
        # X[i] = X_values[sum(X[i]>X_cdf)]
        X[i] = X_values[sum([X[i]>X_cdf[j] for j in range(0, len(X_cdf))])]

    eps = np.random.rand(num_samples)
    for i in range(0, num_samples):
        # eps[i] = noise_values[sum(eps[i]>noise_cdf)]
        eps[i] = noise_values[sum([eps[i] > noise_cdf[j] for j in range(0, len(noise_cdf))])]
    Y = (fct[X.astype(int)] * eps) % num_states_Y

    # experiments
    exp_size = int(num_samples / len(X_values))
    X_int = []
    Y_int = []
    for x_int in X_values:
        eps = np.random.rand(exp_size)
        for i in range(0, exp_size):
            eps[i] = noise_values[sum([eps[i] > noise_cdf[j] for j in range(0, len(noise_cdf))])]
        X_ = np.ones(exp_size)*x_int
        X_int.append(X_)
        Y_int.append((fct[X_.astype(int)] * eps) % num_states_Y)

    X = np.array(np.concatenate([X, np.concatenate(X_int)]))
    Y = np.array(np.concatenate([Y, np.concatenate(Y_int)]))
    return X.astype(int), Y.astype(int)


def print_result(res):
    total = len(res.values())
    both = round(sum([e == 0 for e in res.values()]) / total * 100, 2)
    correct = round(sum([e == 1 for e in res.values()]) / total * 100, 2)
    wrong = round(sum([e == -1 for e in res.values()]) / total * 100, 2)
    no_direction = round(sum([e == 2 for e in res.values()]) / total * 100, 2)
    print(f'{correct} % correct, {wrong} % wrong, {both} % both, {no_direction} % no direction')


def exp1b():
    level = 0.05
    x_states = [2, 3, 3, 5]
    y_states = [2, 3, 5, 3]
    f_counter = dict()
    p_counter = dict()
    n_counter = dict()
    res = dict()
    no_fit =dict()
    for mod in range(0, 1):
        f_counter[mod] = []
        p_counter[mod] = []
        n_counter[mod] = []
        x_st = x_states[mod]
        y_st = y_states[mod]
        for i in range(0, 1000):
            print(i)
            x_rand = np.array([0])
            x_rand = np.append(x_rand, np.sort(np.random.rand(x_st - 1)))
            x_rand = np.append(x_rand, np.array([1]))
            n_rand = np.array([0])
            n_rand = np.append(n_rand, np.sort(np.random.rand(y_st - 1)))
            n_rand = np.append(n_rand, np.array([1]))
            x_distr = np.diff(x_rand)
            n_distr = np.diff(n_rand)
            fct_rand = np.random.randint(0, y_st, size=x_st)
            # fct_rand = randi([0,y_st-1],x_st,1)
            while sum(abs(np.diff(fct_rand)))==0:
                fct_rand = np.random.randint(0, y_st, size=x_st)
            X, Y = add_noise_cyclic(10000, fct_rand, x_distr, n_distr)

            data = pd.DataFrame({'X': X, 'Y': Y})
            if not os.path.exists(f'../simulations/add_noise_1b/{x_states[mod]}_{y_states[mod]}'):
                os.makedirs(f'../simulations/add_noise_1b/{x_states[mod]}_{y_states[mod]}')
            filename = f'pair{i}.csv'
            data.to_csv(f'../simulations/add_noise_1b/{x_states[mod]}_{y_states[mod]}/{filename}', sep=" ", header=False, index=False)
            fct, p, fct_bw, p_bw = fit_both_dir_discrete(X, True, Y, True, level)
            index = str(mod) + str(i)
            if (p > level) and (p_bw > level):
                res[index] = 0
                no_fit[index] = 1
                f_counter[mod] = f_counter[mod] + [fct_rand]
                n_counter[mod] = n_counter[mod] + [n_distr]
                p_counter[mod] = p_counter[mod] + [x_distr]
            elif (p > level) and (p_bw < level):
                res[index] = 1
            elif (p < level) and (p_bw > level):
                res[index] = -1
                f_counter[mod] = f_counter[mod] + [fct_rand]
                n_counter[mod] = n_counter[mod] + [n_distr]
                p_counter[mod] = p_counter[mod] + [x_distr]
            elif (p < level) and (p_bw < level):
                res[index] = 2
                no_fit[index] = -1

            print_result(res)
        print(mod)
        print(res)
        f_counter = dict()
        p_counter = dict()
        n_counter = dict()
        res = dict()
        no_fit = dict()
    pass


if __name__ == "__main__":
    exp1b()
