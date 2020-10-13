# %res tells you the results:
# %    0: both directions
# %    1: only correct direction
# %    2: no direction
# %   -1: only wrong direction inferred.
# %
# %f_counter, p_counter and n_counter tell you more about the instances of 0 and -1.
import os
import numpy as np
import pandas as pd
from dr.discrete_regression import fit_both_dir_discrete
from dr.exp1b import add_noise_cyclic, print_result


def exp3b(only_generate: bool):
    level = 0.05

    x_st = 20
    y_st = 3
    # x_st=10; y_st= 2;
    # x_st= 3; y_st=20;
    # x_st=20; y_st= 3;
    f_counter = []
    n_counter = []
    p_counter = []
    res = dict()
    for i in range(0, 1000):
        print(i)
        if only_generate:
            x_rand = np.array([0])
            x_rand = np.append(x_rand, np.sort(np.random.rand(x_st - 1)))
            x_rand = np.append(x_rand, np.array([1]))
            n_rand = np.array([0])
            n_rand = np.append(n_rand, np.sort(np.random.rand(y_st - 1)))
            n_rand = np.append(n_rand, np.array([1]))
            x_distr = np.diff(x_rand)
            n_distr = np.diff(n_rand)
            fct_rand = np.random.randint(0, y_st, size=x_st)
            while sum(abs(np.diff(fct_rand))) == 0:
                fct_rand = np.random.randint(0, y_st, size=x_st)
            X, Y = add_noise_cyclic(500, fct_rand, x_distr, n_distr)
            data = pd.DataFrame({'X': X, 'Y': Y})
            if not os.path.exists(f'../simulations/mult_noise_3b_exp/{x_st}_{y_st}'):
                os.makedirs(f'../simulations/mult_noise_3b_exp/{x_st}_{y_st}')
            filename = f'pair{i}.csv'
            data.to_csv(f'../simulations/mult_noise_3b_exp/{x_st}_{y_st}/{filename}', sep=" ", header=False,
                        index=False)
        else:
            filename = f'pair{i}.csv'
            data = pd.read_csv(f'../simulations/mult_noise_3b_exp/{x_st}_{y_st}/{filename}', sep=" ")
            data.columns = ['X', 'Y']
            X = data['X']
            Y = data['Y']
        if not only_generate:
            fct, p, fct_bw, p_bw = fit_both_dir_discrete(X, True, Y, True, level)
            if (p > level) and (p_bw > level):
                res[i] = 0
                #f_counter = f_counter + [fct_rand]
                #n_counter = n_counter + [n_distr]
                #p_counter = p_counter + [x_distr]
            elif (p > level) and (p_bw < level):
                res[i] = 1
            elif (p < level) and (p_bw > level):
                res[i] = -1
                #f_counter = f_counter + [fct_rand]
                #n_counter = n_counter + [n_distr]
                #p_counter = p_counter + [x_distr]
            elif (p < level) and (p_bw < level):
                res[i] = 2
            print_result(res)
    print_result(res)
    pass


if __name__ == "__main__":
    exp3b(True)
