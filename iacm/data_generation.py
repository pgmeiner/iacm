import numpy as np
import pandas as pd
from scipy.stats import t
from typing import Callable, List


def get_discrete_random_number(sample_size: int, alphabet_size: int) -> np.ndarray:
    return np.random.randint(low=0, high=alphabet_size, size=sample_size).reshape(-1, 1)


def get_random_number(sample_size: int) -> np.array:
    df = np.random.randint(low=2, high=10, size=1)
    r = t.rvs(df, size=sample_size)
    return r


def get_nonlinear_function() -> Callable[[np.array, np.array], np.array]:
    function_index = np.random.randint(low=1, high=4, size=1)
    noise_model = np.random.randint(low=1, high=3, size=1)
    # decides between additive or multiplicative model
    if noise_model == 1:
        if function_index == 1:
            return lambda x, n: np.max(x, 0) + n
        elif function_index == 2:
            return lambda x, n: np.sin(2 * np.pi * x) + n
        elif function_index == 3:
            return lambda x, n: np.sign(x) * np.sqrt(np.abs(x)) + n
    else:
        if function_index == 1:
            return lambda x, n: np.max(x, 0) * n
        elif function_index == 2:
            return lambda x, n: np.sin(2 * np.pi * x) * n
        elif function_index == 3:
            return lambda x, n: np.sign(x) * np.sqrt(np.abs(x)) * n


def get_nonlinear_function_two_args() -> Callable[[np.array, np.array, np.array], np.array]:
    function_index = np.random.randint(low=1, high=4, size=1)
    noise_model = np.random.randint(low=1, high=3, size=1)
    # decides between additive or multiplicative model
    if noise_model == 1:
        if function_index == 1:
            return lambda x, z, n: np.max(np.max(x, z), 0) + n
        elif function_index == 2:
            return lambda x, z, n: np.sin(2 * np.pi * x * z) + n
        elif function_index == 3:
            return lambda x, z, n: np.sign(x) * np.sqrt(np.abs(x))*np.sign(z) * np.sqrt(np.abs(z)) + n
    else:
        if function_index == 1:
            return lambda x, z, n: np.max(np.max(x, z), 0) * n
        elif function_index == 2:
            return lambda x, z, n: np.sin(2 * np.pi * x * z) * n
        elif function_index == 3:
            return lambda x, z, n: np.sign(x) * np.sqrt(np.abs(x)) * np.sign(z) * np.sqrt(np.abs(z)) * n


def get_linear_function() -> Callable[[np.array, np.array], np.array]:
    alpha = np.random.normal(0, 10, 1)
    noise_model = np.random.randint(low=1, high=3, size=1)
    # decides between additive or multiplicative model
    if noise_model == 1:
        return lambda x, n: alpha * x + n
    else:
        return lambda x, n: alpha * x * n


def get_linear_function_two_args() -> Callable[[np.array, np.array, np.array], np.array]:
    alpha = np.random.normal(0, 10, 1)
    beta = np.random.normal(0, 10, 1)
    noise_model = np.random.randint(low=1, high=3, size=1)
    # decides between additive or multiplicative model
    if noise_model == 1:
        return lambda x, z, n: alpha * x + beta * z + n
    else:
        return lambda x, z, n: (alpha * x + beta * z) * n


def get_random_integer_excluding_integer_set(max_number: int, exclusion_set: List[int]):
    next_rand = np.random.randint(low=0, high=max_number, size=1)[0]
    while (next_rand in exclusion_set):
        next_rand = np.random.randint(low=0, high=max_number, size=1)[0]

    return next_rand


def get_random_cyclic_function(alphabet_size_x: int, alphabet_size_y: int) -> Callable[[np.array, np.array], np.array]:
    f = {}
    for x in range(0, alphabet_size_x):
        f[x] = get_random_integer_excluding_integer_set(alphabet_size_y, exclusion_set=[v for v in f.values()])

    noise_model = 1# np.random.randint(low=1, high=3, size=1)[0]
    # decides between additive or multiplicative model
    if noise_model == 1:
        return lambda x_1, n: [((f[i[0]] + n[i][0]) % alphabet_size_y) for i in x_1.tolist()]
    else:
        return lambda x_1, n: [((f[i[0]] * n[i][0]) % alphabet_size_y) for i in x_1.tolist()]


def generate_discrete_data(structure: str, sample_size: int, alphabet_size_x: int,
                           alphabet_size_y: int) -> pd.DataFrame:
    """
    Function that generates samples of discrete data following an underlying causal model X->Y.
    :param structure: specifies the function used for generating Y. Can be either 'linear' or 'nonlinear'.
    :param sample_size: number of samples to generate.
    :param alphabet_size_x: range size of random variable X.
    :param alphabet_size_y: range size of random variable Y.
    :return: pandas dataframe with columns 'X', 'Y' containing the generated sample data.
    """
    if 'nonlinear' in structure:
        f = get_nonlinear_function()
    else:
        f = get_linear_function()

    nb_intervention_samples = int(sample_size / alphabet_size_x)
    nb_samples = nb_intervention_samples * alphabet_size_x
    obs_x = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_x)
    obs_y = np.array(
        [int(el[0]) for el in
         (f(obs_x, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
            reshape(-1, 1).tolist()]).reshape(-1, 1)

    # intervention data
    int_x = np.array(np.concatenate(
        [np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_x)])).reshape(-1, 1)
    int_y = np.array(
        [int(el[0]) for el in
         (f(int_x, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
            reshape(-1, 1).tolist()]).reshape(-1, 1)

    return pd.DataFrame(
        {'X': [el[0] for el in np.concatenate([obs_x, int_x])], 'Y': [el[0] for el in np.concatenate([obs_y, int_y])]})


def generate_continuous_data(structure: str, sample_size: int) -> pd.DataFrame:
    """
    Function that generates samples of continuous data following an underlying causal model X->Y.
    :param structure: specifies the function used for generating Y. Can be either 'linear' or 'nonlinear'.
    :param sample_size: number of samples to generate.
    :return: pandas dataframe with columns 'X', 'Y' containing the generated sample data.
    """
    if 'nonlinear' in structure:
        f = get_nonlinear_function()
    else:
        f = get_linear_function()

    nb_interventions = np.random.randint(low=2, high=10, size=1)
    nb_intervention_samples = int(sample_size / nb_interventions)
    nb_samples = nb_intervention_samples * nb_interventions
    obs_x = get_random_number(sample_size=nb_samples)
    obs_y = f(obs_x, get_random_number(nb_samples)).reshape(1, -1)

    # intervention data
    int_x = np.concatenate([np.repeat(inter, nb_intervention_samples).tolist() for inter in
                            get_random_number(sample_size=nb_interventions).tolist()])
    int_y = f(int_x, get_random_number(nb_samples)).reshape(1, -1)

    return pd.DataFrame({'X': np.concatenate([obs_x.reshape(nb_samples, ), int_x.reshape(nb_samples, )]),
                         'Y': np.concatenate([obs_y.reshape(nb_samples, ), int_y.reshape(nb_samples, )])})


def generate_discrete_data_confounded(structure: str, sample_size: int, alphabet_size_x: int, alphabet_size_y: int,
                                      alphabet_size_z: int) -> pd.DataFrame:
    """
    Function that generates samples of discrete data following an underlying causal model X->Y.
    :param structure: specifies the function used for generating Y. Can be either 'linear' or 'nonlinear'.
    :param sample_size: number of samples to generate.
    :param alphabet_size_x: range size of random variable X.
    :param alphabet_size_y: range size of random variable Y.
    :param alphabet_size_z: range size of random variable Z
    :return: pandas dataframe with columns 'X', 'Y', 'Z' containing the generated sample data.
    """
    if 'nonlinear' in structure:
        f_x = get_nonlinear_function()
        f_y = get_nonlinear_function()
    else:
        f_x = get_linear_function()
        f_y = get_linear_function()

    nb_intervention_samples = int(sample_size / alphabet_size_z)
    nb_samples = nb_intervention_samples * alphabet_size_z
    obs_z = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_z)
    obs_x = np.array(
        [int(el[0]) for el in
         (f_x(obs_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_x)) % alphabet_size_x).
            reshape(-1, 1).tolist()]).reshape(-1, 1)
    obs_y = np.array(
        [int(el[0]) for el in
         (f_y(obs_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
             reshape(-1, 1).tolist()]).reshape(-1, 1)

    # intervention data
    int_z = np.array(np.concatenate(
        [np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_z)])).reshape(-1, 1)
    int_x = np.array(
        [int(el[0]) for el in
         (f_x(int_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_x)) % alphabet_size_x).
            reshape(-1, 1).tolist()]).reshape(-1, 1)
    int_y = np.array(
        [int(el[0]) for el in
         (f_y(int_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
             reshape(-1, 1).tolist()]).reshape(-1, 1)

    return pd.DataFrame(
        {'X': [el[0] for el in np.concatenate([obs_x, int_x])], 'Y': [el[0] for el in np.concatenate([obs_y, int_y])],
         'Z': [el[0] for el in np.concatenate([obs_z, int_z])]})


def generate_discrete_data_confounded2(structure: str, sample_size: int, alphabet_size_x: int, alphabet_size_y: int,
                                      alphabet_size_z: int) -> pd.DataFrame:
    if 'nonlinear' in structure:
        f_x = get_nonlinear_function()
        f_y = get_nonlinear_function()
    else:
        f_x = get_linear_function()
        f_y = get_linear_function()

    nb_intervention_samples = int(sample_size / alphabet_size_z)
    nb_samples = nb_intervention_samples * alphabet_size_z
    obs_z = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_z)
    obs_x = np.array(
        [int(el[0]) for el in
         (f_x(obs_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_x)) % alphabet_size_x).
            reshape(-1, 1).tolist()]).reshape(-1, 1)
    obs_y = np.array(
        [int(el[0]) for el in
         (f_y(obs_x, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
             reshape(-1, 1).tolist()]).reshape(-1, 1)

    # intervention data
    int_z = np.array(np.concatenate(
        [np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_z)])).reshape(-1, 1)
    int_x = np.array(
        [int(el[0]) for el in
         (f_x(int_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_x)) % alphabet_size_x).
            reshape(-1, 1).tolist()]).reshape(-1, 1)
    int_y = np.array(
        [int(el[0]) for el in
         (f_y(int_x, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
             reshape(-1, 1).tolist()]).reshape(-1, 1)

    return pd.DataFrame(
        {'X': [el[0] for el in np.concatenate([obs_x, int_x])], 'Y': [el[0] for el in np.concatenate([obs_y, int_y])],
         'Z': [el[0] for el in np.concatenate([obs_z, int_z])]})

def generate_discrete_data_confounded3(structure: str, sample_size: int, alphabet_size_x: int, alphabet_size_y: int,
                                      alphabet_size_z: int) -> pd.DataFrame:
    if 'nonlinear' in structure:
        f_y = get_nonlinear_function_two_args()
    else:
        f_y = get_linear_function_two_args()

    nb_intervention_samples = int(sample_size / alphabet_size_z)
    nb_samples = nb_intervention_samples * alphabet_size_z
    obs_z = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_z)
    obs_x = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_x)
    obs_y = np.array(
        [int(el[0]) for el in
         (f_y(obs_x, obs_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
             reshape(-1, 1).tolist()]).reshape(-1, 1)

    # intervention data
    int_z = np.array(np.concatenate(
        [np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_z)])).reshape(-1, 1)
    int_x = np.array(np.concatenate(
        [np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_x)])).reshape(-1, 1)
    int_y = np.array(
        [int(el[0]) for el in
         (f_y(int_x, int_z, get_discrete_random_number(sample_size=nb_samples,
                                              alphabet_size=alphabet_size_y)) % alphabet_size_y).
             reshape(-1, 1).tolist()]).reshape(-1, 1)

    return pd.DataFrame(
        {'X': [el[0] for el in np.concatenate([obs_x, int_x])], 'Y': [el[0] for el in np.concatenate([obs_y, int_y])],
         'Z': [el[0] for el in np.concatenate([obs_z, int_z])]})


def generate_discrete_cyclic_data(sample_size: int, alphabet_size_x: int, alphabet_size_y: int) -> pd.DataFrame:
    """
    Function that generates samples of discrete cyclic data following an underlying causal model X->Y with mod alphabet_size_y.
    :param sample_size: number of samples to generate.
    :param alphabet_size_x: range size of random variable X.
    :param alphabet_size_y: range size of random variable Y.
    :return: pandas dataframe with columns 'X', 'Y' containing the generated sample data.
    """
    if alphabet_size_x == alphabet_size_y == 2:
        f = lambda x_1, n: [((i[0] + n[i][0]) % alphabet_size_y) for i in x_1.tolist()]
    else:
        f = get_random_cyclic_function(alphabet_size_x, alphabet_size_y)

    nb_intervention_samples = int(sample_size / alphabet_size_x)
    nb_samples = nb_intervention_samples * alphabet_size_x
    obs_x = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_x)
    n = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_y)
    obs_y = np.array([int(el[0]) for el in f(obs_x, n)]).reshape(-1, 1)

    # intervention data
    int_x = np.array(np.concatenate(
        [np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_x)])).reshape(-1, 1)
    n_int = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_y)
    int_y = np.array([int(el[0]) for el in f(int_x, n_int)]).reshape(-1, 1)

    return pd.DataFrame(
        {'X': [el[0] for el in np.concatenate([obs_x, int_x])], 'Y': [el[0] for el in np.concatenate([obs_y, int_y])],
         'N': [el[0] for el in np.concatenate([n, n_int])]})
